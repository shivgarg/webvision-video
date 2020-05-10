import yaml
import argparse
from transformers import BertConfig
args = argparse.ArgumentParser()
args.add_argument('--data_dir',default='./')
args.add_argument('--stats_file', default='./stats.pkl')
args.add_argument('--video_map_file', default='./videos_map.pkl')
args.add_argument('--ckpt_dir', default='./')
args.add_argument('output_filename')
args=args.parse_args()

# Global dict
global_dict = {}
global_dict['exp_name'] = 'exp1'
global_dict['ckpt_dir'] = args.ckpt_dir
global_dict['base_arch'] = 'bert-small'
global_dict['base_config'] = BertConfig.from_pretrained('bert-base-uncased').to_dict()
# Training config
global_dict['batch_size'] = 8
global_dict['prefetch_size'] = 4
global_dict['lr'] = 0.001
global_dict['epochs'] = 50
global_dict['ckpt_steps'] = 1000


# Dataset Config
dataset = {}
dataset['data_dir'] = args.data_dir
dataset['stats_file'] = args.stats_file
dataset['video_map_file'] = args.video_map_file
dataset['samples_per_instance'] = 3
dataset['sampler'] = 'UniformSampler'

# Head Config
head = {}
head['name'] = 'default'

global_dict['head'] = head
global_dict['dataset'] = dataset

with open(args.output_filename, 'w') as f:
    yaml.dump(global_dict, f)
