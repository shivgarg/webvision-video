import tensorflow as tf
import numpy as np
import pickle
import random
import h5py
import os
import time
import gc
import math

class UniformSampler:
    
    def generator(self):
        # Opening the file
        cfg = self.config
        video_map = self.video_map
        vid_order = list(range(len(video_map)))
        if cfg['shuffle']:
            random.shuffle(vid_order)
        num_vid_data_point = cfg['samples_per_instance']
        if not self.train:
            num_vid_data_point = 1
        idx = 0
        hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
        feat = hf.get('data')[()]
        label = hf.get('labels')[()]
        hf.close()
        cur_file = video_map[vid_order[idx]][0]    
        while idx < len(vid_order) + 1 - num_vid_data_point:
            features = []
            labels = []
            sample_weights = []
            for i in range(num_vid_data_point):
                if len(features) >= 512:
                    break
                if cur_file != video_map[vid_order[idx]][0]:
                    hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
                    feat = hf.get('data')[()]
                    label = hf.get('labels')[()]
                    cur_file = video_map[vid_order[idx]][0]
                    hf.close()
                for k in range(video_map[vid_order[idx]][1],video_map[vid_order[idx]][2]):
                    features.append(feat[k])
                    label_one_hot = np.zeros(513)
                    j = 0
                    cur_weight = 0
                    while (j < len(label[k])) and (label[k][j] != -1):
                        label_one_hot[int(label[k][j])] = 1
                        cur_weight = max(cur_weight, 1./self.freq[int(label[k][j])])
                        j+=1
                    if cfg['weigh_labels']:
                        if j != 0:
                            label_one_hot = label_one_hot/j
                    labels.append(label_one_hot)
                    sample_weights.append([math.sqrt(cur_weight*len(self.video_map))])
                    if len(features) >= 512:
                        break
                idx+=1
            if cfg['weighted']:
                sample_weights = np.array(sample_weights)
                labels = np.concatenate((labels, sample_weights),axis=-1)
        
            yield (features, labels)
        gc.collect()

    def get_spec(self):
        if self.config['weighted']:
            return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None, 514], dtype=tf.float32)], ([None, 2048],[None,514]),(0.,0.))
        else:
            return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None, 513], dtype=tf.float32)], ([None, 2048],[None,513]),(0.,-1.))
    
    def __len__(self):
        return len(self.video_map)

    def get_output_types(self):
            return (tf.dtypes.float32, tf.dtypes.float32)
    
    def get_output_shapes(self):
        if self.config['weighted']:
            return ((None, 2048), (None,514))
        else:
            return ((None,2048), (None,513))

    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        if train:
            self.video_map = pickle.load(open(config['train_video_map_file'],'rb'))
        else:
            self.video_map = pickle.load(open(config['val_video_map_file'],'rb'))
        self.freq = pickle.load(open(config['freq_file'],'rb'))
        print("Num of videos fragments:", len(self.video_map)) 

class UniformSamplerUnique:
    
    def generator(self):
        # Opening the file
        cfg = self.config
        video_map = self.video_map
        vid_order = list(range(len(video_map)))
        if cfg['shuffle']:
            random.shuffle(vid_order)
        num_vid_data_point = cfg['samples_per_instance']
        if not self.train:
            num_vid_data_point = 1
        
        idx = 0
        hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
        feat = hf.get('data')[()]
        label = hf.get('labels')[()]
        cur_file = video_map[vid_order[idx]][0]    
        
        while idx < len(vid_order) + 1 - num_vid_data_point:
            features = []
            labels = []
            sample_weights = []
            for i in range(num_vid_data_point):
                if len(features) >= 512:
                    break
                if cur_file != video_map[vid_order[idx]][0]:
                    hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
                    feat = hf.get('data')[()]
                    label = hf.get('labels')[()]
                    cur_file = video_map[vid_order[idx]][0]
                for j in range(video_map[vid_order[idx]][1],video_map[vid_order[idx]][2]):
                    features.append(feat[j])
                    labels.append([label[j][0]])
                    sample_weights.append([math.sqrt(len(self.video_map)/self.freq[int(label[j][0])])])
                    if len(features) >= 512:
                        break      
                idx += 1
            if cfg['weighted']:
                labels = np.concatenate((labels,sample_weights), axis=-1)
            yield (features, labels)

    def get_spec(self):
        if self.config['weighted']:
            return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32)],([None, 2048],[None, 2]),(0.,0.))
        else:
            return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)],([None, 2048],[None, 1]),(0.,-1.))
    
    def __len__(self):
        return len(self.video_map)

    def get_output_types(self):
        return (tf.dtypes.float32, tf.dtypes.float32)
    
    def get_output_shapes(self):
        if self.config['weighted']:
            return ((None, 2048), (None, 2))
        else:
            return ((None,2048), (None, 1))
    
    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        if train:
            self.video_map = pickle.load(open(config['train_video_map_file'],'rb'))
        else:
            self.video_map = pickle.load(open(config['val_video_map_file'],'rb'))
        self.freq = pickle.load(open(config['freq_file'],'rb'))
        print("Num of videos fragments:", len(self.video_map)) 


"""        
return tf.data.Dataset.from_generator(
            cls.generator,
            output_types=(tf.dtypes.float32, tf.dtypes.int32),
            output_shapes=((None,2048), (None))
        )

dataset = {}
dataset['data_dir'] = '/home/shivam/Documents/cvpr2020/webvision-video/data/train'
dataset['stats_file'] = '/home/shivam/Documents/cvpr2020/webvision-video/configs/stats.pkl'
dataset['train_video_map_file'] = '/home/shivam/Documents/cvpr2020/webvision-video/configs/videos_map_train.pkl'
dataset['val_video_map_file'] = '/home/shivam/Documents/cvpr2020/webvision-video/configs/videos_map_val.pkl'
dataset['samples_per_instance'] = 2
dataset['sampler'] = 'UniformSampler'
dataset['shuffle'] = True
dataset = UniformSamplerUnique(dataset,train=False)
data = tf.data.Dataset.from_generator(dataset.generator, output_types=dataset.get_output_types(),output_shapes=dataset.get_output_shapes())
data = data.padded_batch(1,padded_shapes=([None, 2048],[None])).prefetch(2)
for sample in data:
    print(sample[0].shape, sample[1].shape)
    print(sample[1])

"""
