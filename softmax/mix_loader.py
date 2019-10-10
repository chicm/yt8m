import os, cv2, glob
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.utils import shuffle
import random
import math
from tqdm import tqdm
import settings
from utils import get_classes_1001, get_classes_1000, dequantize


classes, stoi = get_classes_1001()

META_DIR = settings.META_DIR
NUM_CLASSES = 1001

error_count = 0
# df_val = df_val[['vid', 'start_time', 'end_time', 'label', 'score']]
class MixDataset(data.Dataset):
    def __init__(self, df, test_mode=False):
        self.df = df
        self.test_mode = test_mode

    def get_features(self, row):
        #print(row)
        vid = row.vid
        if self.test_mode:
            fn = osp.join(settings.TEST_NPY_DIR, vid+'.npy')
        elif row.flag == 'val':
            fn = osp.join(settings.VAL_NPY_DIR, vid+'.npy')
        elif row.flag == 'train':
            fn = osp.join(settings.TRAIN_NPY_DIR, vid+'.npy')
        elif row.flag == 'neg':
            fn = osp.join(settings.TRAIN_NEG_DIR, vid+'.npy')
        else:
            raise ValueError('img_dir')
        x = np.load(fn, allow_pickle=True).item()

        if self.test_mode or row.flag == 'val':
            start_frame = row.start_time
        else:
            if row.nframes >= 5:
                start_frame = random.randint(0, row.nframes-5)
            else:
                start_frame = 0


        rgb_frames = x['rgb_frame'][start_frame: start_frame+5]
        audio_frames = x['audio_frame'][start_frame: start_frame+5]

        return dequantize(torch.tensor(rgb_frames).float()), dequantize(torch.tensor(audio_frames).float())

    def __getitem__(self, index):
        global error_count

        row = self.df.iloc[index]
        rgb_frames, audio_frames = self.get_features(row)
        #print(rgb_frames.size())
        if rgb_frames.size() != (5, 1024):
            error_count += 1
            #print(error_count)
            #print(rgb_frames.size())
            if self.test_mode:
                raise ValueError('error size')
            else:
                return torch.randn(5,1024), torch.randn(5, 128), 0
            
        if self.test_mode:
            return rgb_frames, audio_frames
        else:
            return rgb_frames, audio_frames, stoi[row.label]

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        rgb_tensor = torch.stack([x[0] for x in batch])
        audio_tensor = torch.stack([x[1] for x in batch])

        if self.test_mode:
            return rgb_tensor, audio_tensor
        else:
            labels = torch.tensor([x[2] for x in batch])
            return rgb_tensor, audio_tensor, labels

N_FRAMES = 5
import random
class FrameDataset(data.Dataset):
    def __init__(self, df):
        self.df = df

    def get_features(self, row):
        #print(row)
        vid = row.vid
        fn = osp.join(settings.TRAIN_NPY_DIR, vid+'.npy')
        x = np.load(fn, allow_pickle=True).item()

        if row.nframes > N_FRAMES:
            start_frame = random.randint(0, row.nframes-N_FRAMES)
        else:
            start_frame = 0

        rgb_frames = x['rgb_frame'][start_frame:start_frame+N_FRAMES]
        audio_frames = x['audio_frame'][start_frame:start_frame+N_FRAMES]

        return dequantize(torch.tensor(rgb_frames).float()), dequantize(torch.tensor(audio_frames).float())

    def __getitem__(self, index):
        global error_count

        row = self.df.iloc[index]
        rgb_frames, audio_frames = self.get_features(row)
        #print(row)

        return rgb_frames, audio_frames, stoi[str(row.label)]

    def __len__(self):
        return len(self.df)

    def _pad_sequence_tensor(self, batch):
        seq_lens = [len(x) for x in batch]
        max_seq_len = max(seq_lens)
        masks = [[1]*x + [0]*(max_seq_len-x) for x in seq_lens]
        num_seq = len(batch)
        vector_len = len(batch[0][0])
        #print(vector_len)
        new_batch = torch.zeros(num_seq, max_seq_len, vector_len)
        for i, seq in enumerate(batch):
            for j, w in enumerate(seq):
                new_batch[i, j] = w
        return new_batch, torch.tensor(masks)

    def collate_fn(self, batch):
        rgb_tensor, rgb_masks = self._pad_sequence_tensor([x[0] for x in batch])
        audio_tensor, audio_masks = self._pad_sequence_tensor([x[1] for x in batch])
        
        labels = torch.tensor([x[2] for x in batch])
        #return rgb_tensor, rgb_masks, audio_tensor, audio_masks, labels
        return rgb_tensor, audio_tensor, labels


def prepare_train_df(val_split=0.8):
    df_train = pd.read_csv(osp.join(settings.META_DIR, 'train_all.csv'), converters={'label': eval})
    df_train.label = df_train.label.map(lambda x: str(x[0]))
    df_train['flag'] = 'train'
    df_train['start_time'] = 0
    df_train['end_time'] = 0
    df_train['score'] = 0

    df_neg = pd.read_csv(osp.join(settings.META_DIR, 'train_neg.csv'))
    df_neg['flag'] = 'neg'
    df_neg['start_time'] = 0
    df_neg['end_time'] = 0
    df_neg['score'] = 0
    
    df_val = pd.read_csv(osp.join(settings.META_DIR, 'val.csv'))
    df_val['nframes'] = 0
    df_val['flag'] = 'val'
    split_index = int(len(df_val) * val_split)

    df_all = shuffle(pd.concat([df_train, df_neg] + [df_val.iloc[:split_index].copy()]*5, sort=False))

    return df_all, df_val.iloc[split_index:].copy()

def get_train_val_loaders(batch_size=4, val_batch_size=4, val_percent=0.9, dev_mode=False):
    df_train, df_val = prepare_train_df()
    #df_val = df_val.loc[df_val.label!='none'].copy()
    if dev_mode:
        df_train = df_train.iloc[:100]
        df_val = df_val.iloc[:100]
    print('train:', df_train.shape, 'val:', df_val.shape)

    train_ds = MixDataset(df_train, test_mode=False)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=24, collate_fn=train_ds.collate_fn, drop_last=True)
    train_loader.num = len(train_ds)
    train_loader.seg = True

    val_ds = MixDataset(df_val, test_mode=False)
    val_loader = data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=24, collate_fn=val_ds.collate_fn, drop_last=False)
    val_loader.num = len(val_ds)
    
    return train_loader, val_loader

def get_test_loader(batch_size=4, dev_mode=False):
    df = pd.read_csv(osp.join(settings.META_DIR, 'test_ids.csv'))
    if dev_mode:
        df = df.iloc[:100]
    test_ds = MixDataset(df, test_mode=True)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=24, collate_fn=test_ds.collate_fn, drop_last=False)
    test_loader.num = len(test_ds)

    return test_loader

def test_train_loader():
    train_loader, val_loader = get_train_val_loaders(dev_mode=False)
    for x1, x2, labels in train_loader:
        print(x1.size(), x2.size())
        print(labels)
        print(x1)
        break

def test_frame_loader():
    frame_loader = get_frame_train_loader()
    for rgb_tensor, rgb_masks, audio_tensor, audio_masks, labels in frame_loader:
        print(rgb_tensor, labels)
        print(rgb_masks)
        print(audio_tensor.size(), rgb_tensor.size())
        break

def test_test_loader():
    test_loader = get_test_loader(4, dev_mode=True)
    for x1, x2 in test_loader:
        print(x1.size(), x2.size())
        print(x2)
        break

def test_mix():
    frame_loader = get_frame_train_loader(dev_mode=True, batch_size=4)
    train_loader, val_loader = get_train_val_loaders(dev_mode=True, batch_size=4)

    def get_batch(loader, iterator=None):
        if iterator is None:
            iterator = loader.__iter__()
        try:
            b = iterator.__next__()
        except StopIteration:
            iterator = loader.__iter__()
            b = iterator.__next__()
        return b, iterator
    
    t1 = frame_loader.__iter__()
    t2 = train_loader.__iter__()
    while True:
        #print(len(frame_loader))
        for i in range(3):
            x, t1 = get_batch(frame_loader, t1)
            print('t1:', x[-1])
        for i in range(1):
            x, t2 = get_batch(train_loader, t2)
            print('t2:', x[-1])
    

if __name__ == '__main__':
    test_train_loader()
    #test_test_loader()
    #test_frame_loader()
    #test_mix()
