from sentence_transformers import SentenceTransformer
import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--labels_file")
args.add_argument("--dest_file")
args = args.parse_args()


model = SentenceTransformer('bert-base-nli-mean-tokens')
labels = open(args.labels_file,'r')
weight_matrix = []
for line in labels:
    line = line.replace('_',' ')
    embed = model.encode([line])
    weight_matrix.append(embed)

weight_matrix = np.squeeze(np.transpose(np.array(weight_matrix)))
print(weight_matrix.shape)
with open(args.dest_file, 'wb') as f:
    np.save(f, weight_matrix)


