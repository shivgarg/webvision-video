import h5py 
import os
import argparse
import pickle
from tqdm import tqdm
import random

#id, filename, start_idx, end_idx
classes = {}
NUM_CLASSES = 513
for i in range(NUM_CLASSES):
    classes[i] = []

args = argparse.ArgumentParser()
args.add_argument('dir')
args.add_argument('dest_dir')
args = args.parse_args()

val_frac = 0.2

videos = []
for d,s,f in os.walk(args.dir):
    for files in tqdm(f):
        hf = h5py.File(os.path.join(d,files),'r')
        filenames = list(hf.get('filenames')[()])
        filenames.append('')
        labels = hf.get('labels')[()]
        last = filenames[0]
        start = 0
        for i,fname in enumerate(filenames[1:]):
            if last != fname:
                for label_idx in labels[i]:
                    if label_idx == -1:
                        break
                    classes[label_idx].append(len(videos))
                videos.append([files, start,i+1,labels[i]])
                start = i + 1
                last = fname

print("Num of videos:",len(videos))

num_videos = len(videos)
videos_list = list(range(num_videos))
random.shuffle(videos_list)

num_val = int(val_frac*num_videos)
val_videos = videos[:num_val]
train_videos = videos[num_val:]

with open(os.path.join(args.dest_dir,'videos_map_train.pkl'),'wb') as f:
    pickle.dump(train_videos, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(args.dest_dir,'videos_map_val.pkl'),'wb') as f:
    pickle.dump(val_videos, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(args.dest_dir,'stats.pkl'),'wb') as f:
    pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)
