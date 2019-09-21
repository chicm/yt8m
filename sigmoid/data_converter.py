import os
import os.path as osp
import glob
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
from utils import get_classes_1000


VAL_DATA_DIR = '/home/chec/data/yt8m/3/frame/validate'
VAL_OUT_DIR = '/home/chec/data/yt8m/3/frame/val_npy'

TEST_DATA_DIR = '/home/chec/data/yt8m/3/frame/test'
TEST_OUT_DIR = '/home/chec/data/yt8m/3/frame/test_npy'

TRAIN_DATA_DIR = '/home/chec/data/yt8m/2/frame/train'
#TRAIN_OUT_DIR = '/home/chec/data/yt8m/2/frame/train_single_npy'
TRAIN_OUT_DIR = '/home/chec/data/yt8m/2/frame/train_npy_all'

classes, stoi = get_classes_1000()

def save_data_as_npy(tf_filename, out_dir, val=True):
    for example in tqdm(tf.python_io.tf_record_iterator(tf_filename)):
        dataset = dict()
        tf_example = tf.train.Example.FromString(example)
        #print(tf_example)
        tf_seq_example = tf.train.SequenceExample.FromString(example)

        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        if val:
            vid_labels = tf_example.features.feature['labels'].int64_list.value  
            segment_labels = tf_example.features.feature['segment_labels'].int64_list.value
            segment_end_times = tf_example.features.feature['segment_end_times'].int64_list.value
            segment_start_times = tf_example.features.feature['segment_start_times'].int64_list.value
            segment_scores = tf_example.features.feature['segment_scores'].float_list.value
            
        rgb_values = []
        audio_values = []
        for i in range(n_frames):
            rgb_values.append(tf_seq_example.feature_lists.feature_list['rgb']
                            .feature[i].bytes_list.value[0])
            audio_values.append(tf_seq_example.feature_lists.feature_list['audio']
                            .feature[i].bytes_list.value[0])
        sess = tf.InteractiveSession()
        rgb_frame = tf.decode_raw(rgb_values, tf.uint8).eval()
        audio_frame = tf.decode_raw(audio_values, tf.uint8).eval()
        sess.close()
        tf.reset_default_graph()
        
        #print('x')
        dataset['id'] = vid_id
        if val:
            dataset['labels'] = list(vid_labels)
            dataset['segment_labels'] = list(segment_labels)
            dataset['segment_end_times'] = list(segment_end_times)
            dataset['segment_scores'] = list(segment_scores)
            dataset['segment_start_times'] = list(segment_start_times)
            
        dataset['rgb_frame'] = list(rgb_frame)
        dataset['audio_frame'] = list(audio_frame)
        #print('rgb frame:', rgb_frame.shape)
        #print('seg end times:', seg_end_times)
        
        out_file = osp.join(out_dir,  vid_id + '.npy')
        assert not os.path.exists(out_file)
        np.save(out_file, np.array(dataset))


def convert_val_data():
    try:
        os.stat(VAL_OUT_DIR)
    except:
        os.mkdir(VAL_OUT_DIR)

    val_files = glob.glob(VAL_DATA_DIR+'/*.tfrecord')
    print('Found {} tfrecord files'.format(len(val_files)))

    for fn in tqdm(val_files):
        save_data_as_npy(fn, VAL_OUT_DIR)


def convert_test_data():
    try:
        os.stat(TEST_OUT_DIR)
    except:
        os.mkdir(TEST_OUT_DIR)

    test_files = glob.glob(TEST_DATA_DIR+'/*.tfrecord')
    print('Found {} tfrecord files'.format(len(test_files)))

    for fn in tqdm(test_files):
        save_data_as_npy(fn, TEST_OUT_DIR, val=False)

def convert_train_data():
    try:
        os.stat(TRAIN_OUT_DIR)
    except:
        os.mkdir(TRAIN_OUT_DIR)

    train_files = glob.glob(TRAIN_DATA_DIR+'/*.tfrecord')
    print('Found {} tfrecord files'.format(len(train_files)))

    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.enable_eager_execution()

    #sess = tf.InteractiveSession()
    meta = []

    for fn in tqdm(train_files):
        save_train_data_as_npy(fn, TRAIN_OUT_DIR, meta)

    df_train = pd.DataFrame(meta)
    df_train = df_train[['vid', 'label', 'nframes']]
    #df_train.head()
    df_train.to_csv('train.csv', index=False)
    #sess.close()
    #tf.reset_default_graph()

 
def save_train_data_as_npy(tf_filename, out_dir, meta):
    for example in tqdm(tf.python_io.tf_record_iterator(tf_filename)):
        dataset = dict()
        tf_example = tf.train.Example.FromString(example)
        tf_seq_example = tf.train.SequenceExample.FromString(example)

        #print(tf_example)
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        vid_labels = tf_example.features.feature['labels'].int64_list.value
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        #print(vid_id, vid_labels, n_frames)

        filtered_labels = list(set(vid_labels) & set(classes))
        #if len(filtered_labels) == 1:
        if filtered_labels:
            rgb_values = []
            audio_values = []
            for i in range(n_frames):
                rgb_values.append(tf_seq_example.feature_lists.feature_list['rgb']
                                .feature[i].bytes_list.value[0])
                audio_values.append(tf_seq_example.feature_lists.feature_list['audio']
                                .feature[i].bytes_list.value[0])
            sess = tf.InteractiveSession()
            #tf.logging.set_verbosity(tf.logging.ERROR)
            rgb_frame = tf.decode_raw(rgb_values, tf.uint8).eval()
            audio_frame = tf.decode_raw(audio_values, tf.uint8).eval()
            sess.close()
            tf.reset_default_graph()

            dataset['id'] = vid_id
            dataset['labels'] = filtered_labels
            dataset['rgb_frame'] = list(rgb_frame)
            dataset['audio_frame'] = list(audio_frame)

            out_file = osp.join(out_dir,  vid_id + '.npy')
            assert not os.path.exists(out_file)
            np.save(out_file, np.array(dataset))

            meta.append({
                'vid': vid_id,
                'label': filtered_labels,
                'nframes': len(list(rgb_frame))
            })


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data converter')
    parser.add_argument('--data', type=str, choices=['val', 'test', 'train'], default='val')

    args = parser.parse_args()
    print(args)

    if args.data == 'val':
        convert_val_data()
    elif args.data == 'train':
        convert_train_data()
    else:
        convert_test_data()
