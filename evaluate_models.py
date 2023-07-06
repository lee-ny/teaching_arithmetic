"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import tqdm
import glob
import re
import numpy as np

from main_utils import load_trained_model, evaluate_addition, get_encode_decode, evaluate_text

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-addition-char-pad' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
ckpt_path_name = 'ckpt.pt'
dataset = 'shakespeare_addition_char'
batch_size = 64
analyze = False

wandb_log=True
wandb_entity = 'aegean-transformers'
wandb_project = 'addition-char'
wandb_run_name = 'num_train-mini-gpt-padded'

plot_sample_acc = True
eval_addition = True
eval_text = True
eval_text_data_path = 'data/shakespeare_addition_char/val_shakespeare.bin' 

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# look for the meta pickle in case it is available in the dataset folder
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    meta_path = None
encode, decode = get_encode_decode(meta_path)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)

ckpt_list = glob.glob(out_dir+'/ckpt_*_final.pt')
ckpt_list.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])

for ckpt_path_name in ckpt_list:
    # load model
    print(f'loading: {ckpt_path_name}')
    num_train = int(''.join(x for x in ckpt_path_name if x.isdigit()))

    checkpoint = torch.load(ckpt_path_name, map_location=device)
    model = load_trained_model(config, checkpoint)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # evaluate addition performance
    if eval_addition:
        test_accuracy, accuracy_dict = evaluate_addition(config, model, ctx, encode, decode, verbose=True, analyze=analyze)
    if eval_text:
        eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
        ppl = evaluate_text(config, model, eval_text_data, ctx)
    if wandb_log:
        wandb_dict={"num_train_samples": num_train, 
        "test_accuracy": test_accuracy if eval_addition else None,
        "test_perplexity": ppl if eval_text else None
        }
        wandb_dict.update(accuracy_dict)
        wandb.log(wandb_dict)