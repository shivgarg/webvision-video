import tensorflow as tf
import pickle
import random
import h5py
import os

class UniformSampler:
    
    config = {}
    
    @classmethod
    def generator(cls):
        # Opening the file
        cfg = cls.config
        video_map = pickle.load(open(cfg['video_map_file'],'rb'))
        vid_order = list(range(len(video_map)))
        random.shuffle(vid_order)
        num_vid_data_point = cfg['samples_per_instance']
        idx = 0
        while idx < len(vid_order) + 1 - num_vid_data_point:
            features = []
            labels = []
            for i in range(num_vid_data_point):
                hf = h5py.File(os.path.join(cfg['data_dir'],video_map[idx+i][0]),'r') 
                feat = hf.get('data')[()]
                label = hf.get('labels')[()]
                for i in range(video_map[idx+i][1],video_map[idx+i][2]):
                    features.append(feat[i])
                    label_one_hot = [0 for i in range(513)]
                    j=0
                    while label[i][j] != -1:
                        label_one_hot[int(label[i][j])] = 1
                        j+=1
                    labels.append(label_one_hot)
                
            # Reading data (line, record) from the file
            idx+=2
            yield (features, labels)

    def __new__(cls, config):
        cls.config = config
        return tf.data.Dataset.from_generator(
            cls.generator,
            output_types=(tf.dtypes.float32, tf.dtypes.int64),
            output_shapes=((None,2048), (None,513))
        )

"""
dataset = {}
dataset['data_dir'] = '/home/shivam/Documents/cvpr2020/webvision-video/data/train'
dataset['stats_file'] = '/home/shivam/Documents/cvpr2020/webvision-video/configs/stats.pkl'
dataset['video_map_file'] = '/home/shivam/Documents/cvpr2020/webvision-video/configs/videos_map.pkl'
dataset['samples_per_instance'] = 2
dataset['sampler'] = 'UniformSampler'

data = UniformSampler(dataset).padded_batch(3,padded_shapes=([None, None],[None,None])).prefetch(2)
for sample in data:
    print(sample[0].shape, sample[1].shape)
    print(len(tf.keras.layers.Masking()(sample[0][0])._keras_mask))
"""