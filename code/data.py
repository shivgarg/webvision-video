import tensorflow as tf
import numpy as np
import pickle
import random
import h5py
import os

class UniformSampler:
    
    def generator(self):
        # Opening the file
        cfg = self.config
        video_map = self.video_map
        vid_order = list(range(len(video_map)))
        if cfg['shuffle']:
            random.shuffle(vid_order)
        num_vid_data_point = cfg['samples_per_instance']
        idx = 0
        while idx < len(vid_order) + 1 - num_vid_data_point:
            features = []
            labels = []
            for i in range(num_vid_data_point):
                hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
                feat = hf.get('data')[()]
                label = hf.get('labels')[()]
                if len(features) >= 512:
                    break
                for k in range(video_map[vid_order[idx]][1],video_map[vid_order[idx]][2]):
                    features.append(feat[k])
                    if len(features) >= 512:
                        break
                    label_one_hot = np.zeros(513)
                    j=0
                    while (j < len(label[k])) and (label[k][j] != -1):
                        label_one_hot[int(label[k][j])] = 1
                        j+=1
                    if cfg['weigh_labels']:
                        if j!=0:
                            label_one_hot = label_one_hot/j
                    labels.append(label_one_hot)
                
                idx+=1
            yield (features, labels)

    def get_spec(self):
        return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None, 513], dtype=tf.float32)], ([None, 2048],[None,513]),(0.,-1.))
    
    def get_len(self):
        return len(self.video_map)

    def get_output_types(self):
        return (tf.dtypes.float32, tf.dtypes.float32)
    
    def get_output_shapes(self):
        return ((None,2048), (None))

    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        if train:
            self.video_map = pickle.load(open(config['train_video_map_file'],'rb'))
        else:
            self.video_map = pickle.load(open(config['val_video_map_file'],'rb'))
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
        while idx < len(vid_order) + 1 - num_vid_data_point:
            features = []
            labels = []
            for i in range(num_vid_data_point):
                if len(features) >= 512:
                    break
                hf = h5py.File(os.path.join(cfg['data_dir'],video_map[vid_order[idx]][0]),'r') 
                feat = hf.get('data')[()]
                label = hf.get('labels')[()]
                for j in range(video_map[vid_order[idx]][1],video_map[vid_order[idx]][2]):
                    features.append(feat[j])
                    labels.append(label[j][0])
                    if len(features) >= 512:
                        break      
                idx += 1
            yield (features, labels)

    def get_spec(self):
        return ([tf.TensorSpec(shape=[None, None, 2048], dtype=tf.float32),tf.TensorSpec(shape=[None, None], dtype=tf.int32)],([None, 2048],[None]),(0.,-1))
    
    def get_len(self):
        return len(self.video_map)

    def get_output_types(self):
        return (tf.dtypes.float32, tf.dtypes.int32)
    
    def get_output_shapes(self):
        return ((None,2048), (None))
    
    def get_pad_val(self):
        return (0.,-1)

    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        if train:
            self.video_map = pickle.load(open(config['train_video_map_file'],'rb'))
        else:
            self.video_map = pickle.load(open(config['val_video_map_file'],'rb'))
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