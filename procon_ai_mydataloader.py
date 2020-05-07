import torch, queue
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from fastai.imports import *
#from fastai.core import *
import collections,sys,traceback,threading

from fastai.core import *
string_classes = (str, bytes)

def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor.
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a, async=True)
    return a

def get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin: batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin, half) for sample in batch]

    raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


class MyDataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
        self.transpose,self.transpose_y,self.pad_idx,self.half = transpose,transpose_y,pad_idx,half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self): return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b)==ml: return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i,o in enumerate(b):
            if self.pre_pad: res[i, -len(o):] = o
            else:            res[i,  :len(o)] = o
        return res

    def my_jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        if isinstance(b[0][0],(np.ndarray, list)): # list of sentences! newwwwwwwwwww
            mxns = max(len(o) for o in b)  # max no of sentences for each batch
            mins = min(len(o) for o in b)
            mxsl = max(len(s) for d in b for s in d)
            minsl = min(len(s) for d in b for s in d)
            if mxns == mins and mxsl == minsl:
                return np.stack(b) #??
            res = np.zeros((len(b), mxns, mxsl), dtype=np.int64) + self.pad_idx
            for i, d in enumerate(b):
                for j, s in enumerate(d):
                    if self.pre_pad:
                        res[i, j, -len(s):] = s
                    else:
                        res[i, j, :len(s)] = s
            return res

        else:

            ml = max(len(o) for o in b)
            #if min(len(o) for o in b)==ml: return np.stack(b)
            res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
            for i,o in enumerate(b):
                if self.pre_pad: res[i, -len(o):] = o
                else:            res[i, :len(o)] = o
            return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)): return self.my_jag_stack(batch) # changeeeeed
        elif isinstance(b, (int, float)): return np.array(batch)
        elif isinstance(b, string_classes): return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:
            for i in range(len(res)-1): # context, question, label
                if len(res[i].shape)==3:
                    res[i] = res[i].transpose((1,2,0))
                else:
                    res[i] = res[i].T
        if self.transpose_y: res[-1] = res[-1].T
        return res

    def __iter__(self):
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield get_tensor(batch, self.pin_memory, self.half)

