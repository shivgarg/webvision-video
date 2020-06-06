import os
import tensorflow as tf
import yaml
import argparse
import numpy as np
from models import *
from losses import *
from tqdm import tqdm


MODELS = { "bert-small": BertBasic, "lstm": LSTMBasic, "distil-bert": DistilBert,"distil-bert-norm": DistilBertNorm, "mlp": MLP}
LOSS = {"sigmoid": sigmoid_loss, "cross_entropy": cross_entropy,"kl": kl, "kl_weighted": kl_weighted, "cross_entropy_weighted": cross_entropy_weighted}

args = argparse.ArgumentParser()
args.add_argument('config_file')
args.add_argument('test_dir')
args = args.parse_args()

config = yaml.safe_load(open(args.config_file,'r'))

model = MODELS[config['base_arch']](config)
loss_fn = LOSS[config['loss']]()

ckpt = tf.train.Checkpoint(model = model)
manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], 
                    max_to_keep=config['max_to_keep'], 
                    keep_checkpoint_every_n_hours=1)

def eval_step(inputs_embeds):
    output, _ = model(inputs_embeds, training=False)
    output = loss_fn.get_output(output)
    return output

ckpt.restore(manager.latest_checkpoint) 
if manager.latest_checkpoint:
    print("restored from {}".format(manager.latest_checkpoint))
else:
    print("Initialising from scratch")


save_dir = "{}/{}".format(config['ckpt_dir'], manager.latest_checkpoint.strip('/').strip('.').split('/')[-1])
print("Saving output to {}".format(save_dir))
os.makedirs(save_dir,exist_ok=True)
NUM_FRAMES = 300

for files in tqdm(os.listdir(args.test_dir)):
    video = np.load("{}/{}".format(args.test_dir,files))
    num_frames = video.shape[0]
    cur_frame = 0
    output = []
    while cur_frame < num_frames:
        next_frame = min(cur_frame+NUM_FRAMES,num_frames)
        segment = video[cur_frame:next_frame,:]
        segment = tf.convert_to_tensor(segment)
        segment = tf.expand_dims(segment,0)
        output.append(tf.squeeze(eval_step(segment)).numpy())
        cur_frame = next_frame
    output = np.concatenate(output,axis=0)
    dest_file = "{}/{}".format(save_dir,files)
    with open(dest_file,'wb') as f:
        np.save(f, output)


