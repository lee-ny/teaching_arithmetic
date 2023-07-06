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
import pandas as pd
import yaml

from main_utils import load_trained_model,  get_encode_decode, evaluate_addition_new, evaluate_addition_batch, get_data_list, generate_data_str, create_meta_file, set_seed, get_results_dir

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
ckpt_path_name = 'ckpt_final.pt'
dataset = 'addition_pad'
batch_size = 64
num_digit = 3
analyze = False # set True to analyze the result on different error metric
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
verbose = False
add_space = False
verbose_correct=False
eps = 0.0
# metric_type = 'all' # 'all' or 'accuracy' or 'mse' or 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count'

simple=False
random_A=False
random_C=False

prompt_overall = ''
prompt1 = ''
prompt2 = ''
prompt3 = ''
prompt4 = ''

wandb_log=True
wandb_entity = 'ssdd'
wandb_project = 'addition'
wandb_run_name = 'num_train-mini-gpt-padded'
exp_name = 'default_exp_name'

data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*', 'sin', 'sqrt'
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = False
data_shuffle = True

model_type = 'gpt2' # gpt2 or lstm

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# this is probably overkill but seed everything agian
set_seed(1337)

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def save_this_result_dict(this_result_dict, result_dir, i=0):
    result_df = pd.DataFrame(this_result_dict)
    result_df.to_csv(os.path.join(result_dir, f'result{i}.csv'), index=False)
    print(this_result_dict)
    print('result saved to: ', os.path.join(result_dir, f'result{i}.csv'))

# look for the meta pickle in case it is available in the dataset folder
if data_type == 'binary':
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = None
else:
     # check for data_format
    if data_type == 'text':
        if (data_format == 'reverse' and not reverse_c) or (reverse_c and data_format != 'reverse'):
            raise ValueError('reverse_c must be True for data_format == "reverse"')
        elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
            raise ValueError('algo_reason must be True for data_format == "algo_reasoning"')
    
    data_dir = os.path.join('data', dataset)
    # create meta file on the fly
    test_data_path = prompt_overall[5:]
    test_data_list = get_data_list(test_data_path, operator=operator)
    test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=data_shuffle, simple=simple, random_A=random_A, random_C=random_C)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=test_data_str)

encode, decode = get_encode_decode(meta_path)
result_dir = get_results_dir(config)
config['result_dir'] = result_dir
with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)

ckpt_name = os.path.join(out_dir, ckpt_path_name)

# load model
print(f'loading: {ckpt_name}')

checkpoint = torch.load(ckpt_name, map_location=device)
model = load_trained_model(config, checkpoint)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

overall_accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = 0, 0, 0, 0, 0
result_dict = {'accuracy':[], 'carry0':[], 'carry1':[], 'carry2':[], 'carry3':[]}
config['start'] = prompt_overall
overall_accuracy, accuracy_dictionary = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                              reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                              binary=False, fewshot=False, data_type=data_type, operator=operator, data_format=data_format, verbose_correct=verbose_correct, analyze=analyze)
result_dict['accuracy'].append(overall_accuracy)
result_dict['carry0'].append(accuracy_dictionary['carry0']); result_dict['carry1'].append(accuracy_dictionary['carry1']); result_dict['carry2'].append(accuracy_dictionary['carry2']); result_dict['carry3'].append(accuracy_dictionary['carry3'])

if prompt1 != '':
    config['start'] = prompt1
    accuracy1, accuracy_dictionary = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                           reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                           binary=False, fewshot=False, data_type=data_type, operator=operator, data_format=data_format, verbose_correct=verbose_correct, analyze=analyze)
    result_dict['accuracy'].append(accuracy1)
    result_dict['carry0'].append(accuracy_dictionary['carry0']); result_dict['carry1'].append(accuracy_dictionary['carry1']); result_dict['carry2'].append(accuracy_dictionary['carry2']); result_dict['carry3'].append(accuracy_dictionary['carry3'])
    
    this_result_dict = {'prompt':prompt1, 'accuracy':accuracy1, 'carry0':accuracy_dictionary['carry0'], 'carry1':accuracy_dictionary['carry1'], 'carry2':accuracy_dictionary['carry2'], 'carry3':accuracy_dictionary['carry3']}
    if analyze:
        this_result_dict.update(accuracy_dictionary)
    save_this_result_dict(this_result_dict, result_dir, i=1)

if prompt2 != '':
    config['start'] = prompt2
    accuracy2, accuracy_dictionary = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                           reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                           binary=False, fewshot=False, data_type=data_type, operator=operator, data_format=data_format, verbose_correct=verbose_correct, analyze=analyze)
    result_dict['accuracy'].append(accuracy2)
    result_dict['carry0'].append(accuracy_dictionary['carry0']); result_dict['carry1'].append(accuracy_dictionary['carry1']); result_dict['carry2'].append(accuracy_dictionary['carry2']); result_dict['carry3'].append(accuracy_dictionary['carry3'])

    this_result_dict = {'prompt':prompt2, 'accuracy':accuracy2, 'carry0':accuracy_dictionary['carry0'], 'carry1':accuracy_dictionary['carry1'], 'carry2':accuracy_dictionary['carry2'], 'carry3':accuracy_dictionary['carry3']}
    if analyze:
        this_result_dict.update(accuracy_dictionary)
    save_this_result_dict(this_result_dict, result_dir, i=2)

if prompt3 != '':
    config['start'] = prompt3
    accuracy3, accuracy_dictionary = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                           reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                           binary=False, fewshot=False, data_type=data_type, operator=operator, data_format=data_format, verbose_correct=verbose_correct, analyze=analyze)
    result_dict['accuracy'].append(accuracy3)
    result_dict['carry0'].append(accuracy_dictionary['carry0']); result_dict['carry1'].append(accuracy_dictionary['carry1']); result_dict['carry2'].append(accuracy_dictionary['carry2']); result_dict['carry3'].append(accuracy_dictionary['carry3'])

    this_result_dict = {'prompt':prompt3, 'accuracy':accuracy3, 'carry0':accuracy_dictionary['carry0'], 'carry1':accuracy_dictionary['carry1'], 'carry2':accuracy_dictionary['carry2'], 'carry3':accuracy_dictionary['carry3']}
    if analyze:
        this_result_dict.update(accuracy_dictionary)
    save_this_result_dict(this_result_dict, result_dir, i=3)

if prompt4 != '':
    config['start'] = prompt4
    accuracy4, accuracy_dictionary = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                           reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                           binary=False, fewshot=False, data_type=data_type, operator=operator, data_format=data_format, verbose_correct=verbose_correct, analyze=analyze)
    result_dict['accuracy'].append(accuracy4)
    result_dict['carry0'].append(accuracy_dictionary['carry0']); result_dict['carry1'].append(accuracy_dictionary['carry1']); result_dict['carry2'].append(accuracy_dictionary['carry2']); result_dict['carry3'].append(accuracy_dictionary['carry3'])

    this_result_dict = {'prompt':prompt4, 'accuracy':accuracy4, 'carry0':accuracy_dictionary['carry0'], 'carry1':accuracy_dictionary['carry1'], 'carry2':accuracy_dictionary['carry2'], 'carry3':accuracy_dictionary['carry3']}
    if analyze:
        this_result_dict.update(accuracy_dictionary)
    save_this_result_dict(this_result_dict, result_dir, i=4)

if wandb_log:
    wandb_dict = {'overall_accuracy': overall_accuracy, 
        'prompt1_accuracy': accuracy1,
        'prompt2_accuracy': accuracy2,
        'prompt3_accuracy': accuracy3,
        'prompt4_accuracy': accuracy4,
        }
    if analyze:
        wandb_dict.update(accuracy_dictionary)

    wandb.log(wandb_dict)
    result_df = pd.DataFrame(wandb_dict, index=[0])
    result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
    print(wandb_dict)
    print('result saved to: ', os.path.join(result_dir, 'result.csv'))

result_df = pd.DataFrame(result_dict)
result_df.to_csv(os.path.join(result_dir, 'result_carry.csv'), index=False)