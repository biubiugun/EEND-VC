import os
import numpy as np
from functools import partial
import torch
from eend.feature import get_input_dim
from eend.pytorch_backend.models import fix_state_dict
from eend.pytorch_backend.models import PadertorchModel
from eend.pytorch_backend.models import TransformerDiarization
from eend.pytorch_backend.transformer import NoamScheduler
from eend.pytorch_backend.diarization_dataset \
    import DiarizationDatasetFromWave, DiarizationDatasetFromFeat
import padertorch as pt
import padertorch.train.optimizer as pt_opt
from eend import feature
from eend import kaldi_data
import yamlargparse

parser = yamlargparse.ArgumentParser(description='training')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('train_data_dir',
                    help='kaldi-style data dir used for training.')
parser.add_argument('valid_data_dir',
                    help='kaldi-style data dir used for validation.')
parser.add_argument('model_save_dir',
                    help='output directory which model file will be saved in.')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--spkv-lab', default='',
                    help='file path of speaker vector with label and\
                    speaker ID conversion table for adaptation')

# The following arguments are set in conf/train.yaml or conf/adapt.yaml
parser.add_argument('--spk-loss-ratio', default=0.03, type=float)
parser.add_argument('--spkv-dim', default=256, type=int,
                    help='dimension of speaker embedding vector')
parser.add_argument('--max-epochs', default=100, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--input-transform', default='logmel23_mn',
                    choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                             'logmel23_mvn', 'logmel23_swn'],
                    help='input transform')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--optimizer', default='noam', type=str)
parser.add_argument('--num-speakers', default=3, type=int)
parser.add_argument('--gradclip', default=5, type=int,
                    help='gradient clipping. if < 0, no clipping')
parser.add_argument('--chunk-size', default=150, type=int,
                    help='number of frames in one utterance')
parser.add_argument('--batchsize', default=64, type=int,
                    help='number of utterances in one batch.\
                    Note that real batchsize = number of gpu *\
                    batchsize-per-gpu * batchsize')
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--context-size', default=7, type=int)
parser.add_argument('--subsampling', default=10, type=int)
parser.add_argument('--frame-size', default=200, type=int)
parser.add_argument('--frame-shift', default=80, type=int)
parser.add_argument('--sampling-rate', default=8000, type=int)
parser.add_argument('--noam-scale', default=1.0, type=float)
parser.add_argument('--noam-warmup-steps', default=25000, type=float)
parser.add_argument('--transformer-encoder-n-heads', default=8, type=int)
parser.add_argument('--transformer-encoder-n-layers', default=6, type=int)
parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--feature-nj', default=100, type=int,
                    help='maximum number of subdirectories to store\
                    featlab_XXXXXXXX.npy')
parser.add_argument('--batchsize-per-gpu', default=16, type=int,
                    help='virtual_minibatch_size in padertorch')
parser.add_argument('--test-run', default=0, type=int, choices=[0, 1],
                    help='padertorch test run switch; 1 is on, 0 is off')

args = parser.parse_args()
print(args)

def _count_frames(data_len, size, step):
    return int((data_len - size + step) / step)


def _gen_frame_indices(data_length, size=2000, step=2000):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size

    if i * step + size < data_length:
        if data_length - (i + 1) * step > 0:
            if i == -1:
                yield (i + 1) * step, data_length
            else:
                yield data_length - size, data_length

class DiarizationDatasetFromWaveTest(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            dtype=np.float32,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            n_speakers=None,
            ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.rate = rate
        self.input_transform = input_transform
        self.n_speakers = n_speakers

        self.chunk_indices = []
        self.data = kaldi_data.KaldiData(self.data_dir)
        self.all_speakers = sorted(self.data.spk2utt.keys())
        self.all_n_speakers = len(self.all_speakers)
        self.all_n_speakers_arr =\
            np.arange(self.all_n_speakers,
                      dtype=np.int64).reshape(self.all_n_speakers, 1)

        # Make chunk indices: filepath, start_frame, end_frame
        # for rec in self.data.wavs:
        #     data_len = int(self.data.reco2dur[rec] * self.rate / frame_shift)
        #     data_len = int(data_len / self.subsampling)
        #     for st, ed in _gen_frame_indices(data_len, chunk_size, chunk_size):
        #         self.chunk_indices.append(
        #             (rec, st * self.subsampling, ed * self.subsampling))
        for i,rec in enumerate(self.data.wavs):
            self.chunk_indices.append([])
            data_len = int(self.data.reco2dur[rec] * self.rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(data_len, chunk_size, chunk_size):
                self.chunk_indices[i].append(
                    (rec, st * self.subsampling, ed * self.subsampling)
                )
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        # rec, st, ed = self.chunk_indices[i]
        # filtered_segments = self.data.segments[rec]
        # # speakers: the value given from data
        # speakers = np.unique(
        #     [self.data.utt2spk[seg['utt']] for seg in filtered_segments]
        #     ).tolist()
        # n_speakers = self.n_speakers
        # if self.n_speakers < len(speakers):
        #     n_speakers = len(speakers)

        # Y, T = feature.get_labeledSTFT(
        #     self.data,
        #     rec,
        #     st,
        #     ed,
        #     self.frame_size,
        #     self.frame_shift,
        #     n_speakers,
        #     )
        # T = T.astype(np.float32)

        # S_arr = -1 * np.ones(n_speakers).astype(np.int64)
        # for seg in filtered_segments:
        #     speaker_index = speakers.index(self.data.utt2spk[seg['utt']])
        #     all_speaker_index = self.all_speakers.index(
        #         self.data.utt2spk[seg['utt']])
        #     S_arr[speaker_index] = all_speaker_index

        # # If T[:, n_speakers - 1] == 0.0, then S_arr[n_speakers - 1] == -1,
        # # so S_arr[n_speakers - 1] is not used for training,
        # # e.g., in the case of training 3-spk model with 2-spk data

        # Y = feature.transform(Y, self.input_transform)
        # Y_spliced = feature.splice(Y, self.context_size)
        # Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        # ilen = np.array(Y_ss.shape[0], dtype=np.int64)

        # return Y_ss, T_ss, S_arr, self.all_n_speakers_arr, ilen
        rec_list = self.chunk_indices[i]
        wav_data_list = []

        for rec, st, ed in rec_list:
            filtered_segments = self.data.segments[rec]
            # speakers: the value given from data
            speakers = np.unique(
                [self.data.utt2spk[seg['utt']] for seg in filtered_segments]
                ).tolist()
            n_speakers = self.n_speakers
            if self.n_speakers < len(speakers):
                n_speakers = len(speakers)

            Y, T = feature.get_labeledSTFT(
                self.data,
                rec,
                st,
                ed,
                self.frame_size,
                self.frame_shift,
                n_speakers,
                )
            T = T.astype(np.float32)

            S_arr = -1 * np.ones(n_speakers).astype(np.int64)
            for seg in filtered_segments:
                speaker_index = speakers.index(self.data.utt2spk[seg['utt']])
                all_speaker_index = self.all_speakers.index(
                    self.data.utt2spk[seg['utt']])
                S_arr[speaker_index] = all_speaker_index

            # If T[:, n_speakers - 1] == 0.0, then S_arr[n_speakers - 1] == -1,
            # so S_arr[n_speakers - 1] is not used for training,
            # e.g., in the case of training 3-spk model with 2-spk data

            Y = feature.transform(Y, self.input_transform)
            Y_spliced = feature.splice(Y, self.context_size)
            Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
            ilen = np.array(Y_ss.shape[0], dtype=np.int64)
            wav_data = {}
            wav_data['xs'] = Y_ss
            wav_data['ts'] = T_ss
            wav_data['ss'] = S_arr
            wav_data['ns'] = self.all_n_speakers_arr
            wav_data['ilens'] = ilen
            wav_data['rec'] = rec
            wav_data_list.append(wav_data)

        return wav_data_list

    def get_allnspk(self):
        return self.all_n_speakers

def collate_fn_ns(batch, n_speakers, spkidx_tbl):
    xs, ts, ss, ns, ilens = list(zip(*batch)) # feature, activity, speaker ID, speaker number, chunksize
    valid_chunk_indices1 = [i for i in range(len(ts))
                            if ts[i].shape[1] == n_speakers] # 3 == n_speakers
    valid_chunk_indices2 = []

    # n_speakers (rec-data) > n_speakers (model)
    invalid_chunk_indices1 = [i for i in range(len(ts))
                              if ts[i].shape[1] > n_speakers]

    ts = list(ts)
    ss = list(ss)
    for i in invalid_chunk_indices1:
        s = np.sum(ts[i], axis=0)
        cs = ts[i].shape[0]
        if len(s[s > 0.5]) <= n_speakers:
            # n_speakers (chunk-data) <= n_speakers (model)
            # update valid_chunk_indices2
            valid_chunk_indices2.append(i)
            idx_arr = np.where(s > 0.5)[0]
            ts[i] = ts[i][:, idx_arr]
            ss[i] = ss[i][idx_arr]
            if len(s[s > 0.5]) < n_speakers:
                # n_speakers (chunk-data) < n_speakers (model)
                # update ts[i] and ss[i]
                n_speakers_real = len(s[s > 0.5])
                zeros_ts = np.zeros((cs, n_speakers), dtype=np.float32)
                zeros_ts[:, :-(n_speakers-n_speakers_real)] = ts[i]
                ts[i] = zeros_ts
                mones_ss = -1 * np.ones((n_speakers,), dtype=np.int64)
                mones_ss[:-(n_speakers-n_speakers_real)] = ss[i]
                ss[i] = mones_ss
            else:
                # n_speakers (chunk-data) == n_speakers (model)
                pass
        else:
            # n_speakers (chunk-data) > n_speakers (model)
            pass

    # valid_chunk_indices: chunk indices using for training
    valid_chunk_indices = sorted(valid_chunk_indices1 + valid_chunk_indices2)

    ilens = np.array(ilens)
    ilens = ilens[valid_chunk_indices]
    ns = np.array(ns)[valid_chunk_indices]
    ss = np.array([ss[i] for i in range(len(ss))
                  if ts[i].shape[1] == n_speakers])
    xs = [xs[i] for i in range(len(xs)) if ts[i].shape[1] == n_speakers]
    ts = [ts[i] for i in range(len(ts)) if ts[i].shape[1] == n_speakers]
    xs = np.array([np.pad(x, [(0, np.max(ilens) - len(x)), (0, 0)],
                          'constant', constant_values=(-1,)) for x in xs])
    ts = np.array([np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)],
                          'constant', constant_values=(+1,)) for t in ts])

    if spkidx_tbl is not None:
        # Update global speaker ID
        all_n_speakers = np.max(spkidx_tbl) + 1
        bs = len(ns)
        ns = np.array([
                np.arange(
                    all_n_speakers,
                    dtype=np.int64
                    ).reshape(all_n_speakers, 1)] * bs)
        ss = np.array([spkidx_tbl[ss[i]] for i in range(len(ss))])

    return (xs, ts, ss, ns, ilens)


def collate_fn(batch):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    ilens = np.array(ilens)
    xs = np.array([np.pad(
        x, [(0, np.max(ilens) - len(x)), (0, 0)],
        'constant', constant_values=(-1,)
        ) for x in xs])
    ts = np.array([np.pad(
        t, [(0, np.max(ilens) - len(t)), (0, 0)],
        'constant', constant_values=(+1,)
        ) for t in ts])
    ss = np.array(ss)
    ns = np.array(ns)

    return (xs, ts, ss, ns, ilens)

def save_feature(args):
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False

    # device = [device_id for device_id in range(torch.cuda.device_count())]
    device = [1]
    print('GPU device {} is used'.format(device))

    train_set = DiarizationDatasetFromWave(
        args.train_data_dir,
        chunk_size=args.chunk_size,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        n_speakers=args.num_speakers,
        )

    # Count n_chunks
    batchsize = args.batchsize * len(device) * \
        args.batchsize_per_gpu
    f = open('{}/batchsize.txt'.format(args.model_save_dir), 'w')
    f.write("{}\n".format(batchsize))
    f.close()
    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batchsize,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=partial(
                collate_fn_ns,
                n_speakers=args.num_speakers,
                spkidx_tbl=None)
            )
    n_chunks = len(trainloader)
    print("n_chunks : {}".format(n_chunks))
    os.makedirs("{}/data/".format(args.model_save_dir), exist_ok=True)
    f = open('{}/data/n_chunks.txt'.format(args.model_save_dir), 'w')
    f.write("{}\n".format(n_chunks))
    f.close()

    if n_chunks % args.feature_nj == 0:
        max_num_per_dir = n_chunks // args.feature_nj
    else:
        max_num_per_dir = n_chunks // args.feature_nj + 1
    print("max_num_per_dir : {}".format(max_num_per_dir))

    # Save featlab_XXXXXXXX.npy and featlab_chunk_indices.txt
    spkidx_tbl = None
    if args.initmodel:
        # adaptation
        npz = np.load(args.spkv_lab)
        spkidx_tbl = npz['arr_2']

    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batchsize,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=partial(
                collate_fn_ns,
                n_speakers=args.num_speakers,
                spkidx_tbl=spkidx_tbl)
            )
    f = open('{}/data/featlab_chunk_indices.txt'.
             format(args.model_save_dir), 'w')
    idx = 0
    digit_num = len(str(args.feature_nj-1))
    fmt = "{}/data/{:0={}}/featlab_{:0=8}.npy"
    for data in trainloader:
        dir_num = idx // max_num_per_dir
        os.makedirs("{}/data/{:0={}}/".
                    format(args.model_save_dir, dir_num, digit_num),
                    exist_ok=True)
        output_npy_path = fmt.format(args.model_save_dir,
                                     dir_num, digit_num, idx)
        print(output_npy_path)
        bs = data[0].shape[0] #batch size
        cs = data[0].shape[1] #chunk size
        # data0 (feature)
        data0 = data[0]
        # data1 (reference speech activity)
        data1 = data[1]
        # data2 (reference speaker ID)
        data2 = np.zeros([bs, cs, data[2].shape[1]], dtype=np.float32)
        for j in range(bs):
            data2[j, :, :] = data[2][j, :]
        # data3 (reference number of all speakers)
        data3 = np.ones([bs, cs, 1], dtype=np.float32) * len(data[3][0])
        # data4 (real chunk size)
        data4 = np.zeros([bs, cs, 1], dtype=np.float32)
        for j in range(bs):
            data4[j, :, :] = data[4][j]
        save_data = np.concatenate((data0,
                                    data1,
                                    data2,
                                    data3,
                                    data4), axis=2)

        np.save(output_npy_path, save_data)
        for j in range(save_data.shape[0]):
            f.write("{} {}\n".format(output_npy_path, j))
        idx += 1
    f.close()

    # Create completion flag
    f = open('{}/data/.done'.format(args.model_save_dir), 'w')
    f.write("")
    f.close()
    print('Finished!')


def collate_fn_test(batch, n_speakers, spkidx_tbl):
    data_list = list(*batch)
    ms = len(batch)
    bs = len(data_list)
    cs = None
    new_data = None
    for data in data_list:
        xs = data['xs']
        ts = data['ts']
        ss = data['ss']
        ns = data['ns']
        ilens = data['ilens']
        

        xs = np.array([np.pad(xs, [(0, np.max(ilens) - len(xs)), (0, 0)],
                          'constant', constant_values=(-1,))])
        ts = np.array([np.pad(ts, [(0, np.max(ilens) - len(ts)), (0, 0)],
                          'constant', constant_values=(+1,))])


        if cs == None:
            cs = ts.shape[1]
            data0 = xs
            data1 = ts
            data2 = np.zeros([1,cs, ts.shape[2]], dtype=np.float32)
            for j in range(cs):
                data2[0, j, :] = ss[:]
            data3 = np.ones([1, cs, 1], dtype=np.float32) * len(ns)
            data4 = np.zeros([1, cs, 1], dtype=np.float32)
            for j in range(cs):
                data4[0, j, :] = ilens

            new_data = np.concatenate((data0,
                                    data1,
                                    data2,
                                    data3,
                                    data4), axis=2)
        else:
            cs = ts.shape[1]
            data0 = xs
            data1 = ts
            data2 = np.zeros([1,cs, ts.shape[2]], dtype=np.float32)
            for j in range(cs):
                data2[0, j, :] = ss[:]
            data3 = np.ones([1, cs, 1], dtype=np.float32) * len(ns)
            data4 = np.zeros([1, cs, 1], dtype=np.float32)
            for j in range(cs):
                data4[0, j, :] = ilens

            tmp_data = np.concatenate((data0,
                                    data1,
                                    data2,
                                    data3,
                                    data4), axis=2)       
            new_data = np.concatenate((new_data, tmp_data), axis=0)                
        
    return new_data

dataset = DiarizationDatasetFromWaveTest(
    args.train_data_dir,
    chunk_size=args.chunk_size,
    context_size=args.context_size,
    input_transform=args.input_transform,
    frame_size=args.frame_size,
    frame_shift=args.frame_shift,
    subsampling=args.subsampling,
    rate=args.sampling_rate,
    n_speakers=args.num_speakers,
)



dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=partial(
                    collate_fn_test,
                    n_speakers=args.num_speakers,
                    spkidx_tbl=None)
)

for data in dataloader:
    tmp = data
