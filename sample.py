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
from main_utils import load_trained_model, get_encode_decode, evaluate_text, get_data_list, generate_data_str, create_meta_file

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
ckpt_path_name = 'ckpt.pt'
dataset = 'addition_pad'

batch_size = 64
evaluate = False
dataset = 'shakespeare_char'
val_data_path = 'val.bin'

plot_sample_ppl = False # evaluate multiple ckpt trained with different number of dataset
wandb_log = False
wandb_entity = 'aegean-transformers'
wandb_project = 'shakespeare-char'
wandb_run_name = 'num_train-mini-gpt-padded'

data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*'
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = False

model_type = 'gpt2' # gpt2 or lstm

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
if data_type == 'binary':
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = None
else:
     # check for data_format            
    data_dir = os.path.join('data', dataset)
    # create meta file on the fly
    if start.startswith('FILE:'):
        test_data_path = start[5:]
        test_data_list = get_data_list(test_data_path, operator=operator)
        test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=False)
    else:
        test_data_str = start
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=test_data_str)
    
encode, decode = get_encode_decode(meta_path)


# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, ckpt_path_name)
    print('resuming from: ', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = load_trained_model(config, checkpoint)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)

if evaluate:
    # run evaluate
    import numpy as np
    test_data = np.memmap(os.path.join(data_dir, val_data_path), dtype=np.uint16, mode='r')
    perplexity = evaluate_text(config, model, test_data, ctx)
else:
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')
