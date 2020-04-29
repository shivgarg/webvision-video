import h5py 
import os
import argparse
import pickle
from tqdm import tqdm

#id, filename, start_idx, end_idx
classes = {}
NUM_CLASSES = 513
for i in range(NUM_CLASSES):
    classes[i] = []

args = argparse.ArgumentParser()
args.add_argument('dir')
args.add_argument('dest_dir')
args = args.parse_args()

videos = []
for d,s,f in tqdm(os.walk(args.dir)):
    for files in f:
        print(files)
        hf = h5py.File(os.path.join(d,files),'r')
        filenames = hf.get('filenames')[()]
        filenames.append('')
        labels = hf.get('labels')[()]
        last = filenames[0]
        start = 0
        for i,fname in tqdm(enumerate(filenames[1:])):
            if last != fname:
                for label_idx in labels[i]:
                    if label_idx == -1:
                        break
                    classes[label_idx].append(len(videos))
                videos.append([fname, start,i+1,labels[i]])
                start = i + 1
                last = fname

with open('videos_map.pkl','wb') as f:
    pickle.dump(videos, f, pickle.HIGHEST_PROTOCOL)
with open('stats.pkl','wb') as f:
    pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)