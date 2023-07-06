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
import pandas as pd
import yaml

from main_utils import load_trained_model, evaluate_addition, get_encode_decode, evaluate_addition_new, evaluate_addition_batch, set_seed, get_data_list, generate_data_str, create_meta_file, get_results_dir

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-addition-char-pad' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
top_k = 1
print("WARNING! Setting top_k=1 for reproducibility")
seed = 1337
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
ckpt_path_name = 'ckpt_final.pt'
dataset = 'addition_pad'
num_add = 0 # number of samples for plain addition
num_ar = 0 # number of sampels for algorithmic reasoning
lr = '1e-3'
analyze = False
num_digit = 3
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
fewshot = False
binary = False
verbose = True
add_space = False
select = 'dataratio' # either dataratio, samplenum, mixed
num_add = 0
num_ar = 0
test_batch_size = 128
eps = 0.0
tokenizer = 'char' # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'

simple=False
random_A=False
random_C=False

plot_sample_acc = False # evaluate multiple ckpt trained with different number of dataset
wandb_log=False
wandb_entity = 'ssdd'
wandb_project = 'addition-char'
wandb_run_name = 'num_train-mini-gpt-padded'
exp_name = 'default_exp_name'

data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*'
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = False

model_type = 'gpt2' # or lstm

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True # cudnn auto-tuner
torch.backends.cudnn.deterministic = False # cudnn auto-tuner
# this is probably overkill but seed everything agian
set_seed(seed)

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
    if data_type == 'text':
        if ('reverse' in data_format and not reverse_c) or (reverse_c and 'reverse' not in data_format):
            raise ValueError('reverse_c must be True for data_format == "reverse"')
        elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
            raise ValueError('algo_reason must be True for data_format == "algo_reasoning"')
    
    data_dir = os.path.join('data', dataset)
    # create meta file on the fly
    test_data_path = start[5:]
    test_data_list = get_data_list(test_data_path, operator=operator)
    test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=False, simple=simple, random_A=random_A, random_C=random_C)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=test_data_str, tokenizer=tokenizer)
    meta_vocab_size = meta['vocab_size']

encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)
result_dict = {'num_train_samples':[], 'test_accuracy':[], 'num_ar': [], 'num_add': [], 'mean_accuracy': [], 'this_accuracy': [], 'at_least_one_correct': []}
result_dir = get_results_dir(config)
config['result_dir'] = result_dir
with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)
if plot_sample_acc:
    if select == 'dataratio':
        ckpt_list = glob.glob(out_dir+'/ckpt_dr*_final.pt')
    elif select == 'samplenum':
        ckpt_list = glob.glob(out_dir+'/ckpt_*0_final.pt')
    elif select == 'mixed':
        if num_ar == 0:
            num_ar = '*'
        if num_add == 0:
            num_add = '*'
        ckpt_list = glob.glob(out_dir+f'/ckpt_ar{num_ar}_add{num_add}_*final.pt')
    ckpt_list.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
else:
    ckpt_list = [os.path.join(out_dir, ckpt_path_name)]

print(ckpt_list)

for ckpt_path_name in ckpt_list:
    # load model
    print(f'loading: {ckpt_path_name}')
    num_train = int(''.join(x for x in ckpt_path_name.split('/')[-1] if x.isdigit())) if plot_sample_acc else None
    if plot_sample_acc and select == 'mixed':
        num_add_train = int(ckpt_path_name.split(out_dir)[1].split('add')[1].split('_')[0])
        num_ar_train = int(ckpt_path_name.split(out_dir)[1].split('ar')[1].split('_')[0])
        print(f'num_add_train: {num_add_train}, num_ar_train: {num_ar_train}')

    checkpoint = torch.load(ckpt_path_name, map_location=device)
    model = load_trained_model(config, checkpoint)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # evaluate addition performance
    # test_accuracy, accuracy_dict = evaluate_addition(config, model, ctx, encode, decode, verbose=True, analyze=analyze)
    import time
    start = time.time()
    test_accuracy, accuracy_dict = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                                           reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                           binary=binary, fewshot=fewshot, data_type=data_type, operator=operator, data_format=data_format)
    # non-batch version
    # test_accuracy, accuracy_dict = evaluate_addition_new(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, binary=binary, fewshot=fewshot)
    end=time.time()
    print("Time taken by evaluation_addition(): {}".format(end-start))

    if plot_sample_acc and wandb_log:
        if select == 'mixed':
            num_add = int(''.join(x for x in ckpt_path_name.split('/')[-1].split('add')[1].split('_')[0] if x.isdigit()))
            num_ar = int(''.join(x for x in ckpt_path_name.split('/')[-1].split('ar')[1].split('_')[0] if x.isdigit()))
            wandb_dict={"num_ar": num_ar, "num_add": num_add, "mean_accuracy": test_accuracy, "this_accuracy": test_accuracy, "at_least_one_correct": test_accuracy}
            result_dict['num_ar'].append(num_ar)
            result_dict['num_add'].append(num_add)
            result_dict['num_train_samples'].append(0)
            result_dict['test_accuracy'].append(test_accuracy)
            result_dict['mean_accuracy'].append(test_accuracy)
            result_dict['this_accuracy'].append(test_accuracy)
            result_dict['at_least_one_correct'].append(test_accuracy)
        else:
            wandb_dict={"num_train_samples": num_train, "test_accuracy": test_accuracy}
            result_dict['num_train_samples'].append(num_train)
            result_dict['num_ar'].append('')
            result_dict['num_add'].append('')
            result_dict['test_accuracy'].append(test_accuracy)
            result_dict['mean_accuracy'].append(test_accuracy)
            result_dict['this_accuracy'].append(test_accuracy)
            result_dict['at_least_one_correct'].append(test_accuracy)
            # {'num_train_samples':[], 'test_accuracy':[], 'num_ar': [], 'num_add': [], 'mean_accuracy': [], 'this_accuracy': [], 'at_least_one_correct': []}
        wandb_dict.update(accuracy_dict)
        wandb.log(wandb_dict)
        
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
        
    elif wandb_log:
        wandb_dict={"test_accuracy": test_accuracy}
        wandb_dict.update(accuracy_dict)
        wandb.log(wandb_dict)
