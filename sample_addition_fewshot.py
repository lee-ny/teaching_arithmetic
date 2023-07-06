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

from main_utils import load_trained_model, evaluate_addition, get_encode_decode, evaluate_addition_new, evaluate_addition_fewshot_batch, set_seed, get_data_list, generate_data_str, create_meta_file, get_results_dir

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-addition-char-pad' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
ckpt_path_name = 'ckpt_final.pt'
dataset = 'addition_pad'
analyze = False
num_digit = 3
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
binary = False
verbose = True
add_space = False
select = 'fewshot' # either dataratio, samplenum, fewshot
multiple_set_per_prompt = False # used for word prompts - multiple set for each phrase category

simple=False
random_A=False
random_C=False

fewshot=True
prompt_dir = 'prompts/+/few_shot_prefix_ar/1shot_1.txt'

num_ar=0
num_add=0
lr = '1e-3'

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
        if (data_format == 'reverse' and not reverse_c) or (reverse_c and data_format != 'reverse'):
            raise ValueError('reverse_c must be True for data_format == "reverse"')
        elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
            raise ValueError('algo_reason must be True for data_format == "algo_reasoning"')
    
    data_dir = os.path.join('data', dataset)
    # create meta file on the fly
    test_data_path = start[5:]
    test_data_list = get_data_list(test_data_path, operator=operator)
    test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=False, simple=simple, random_A=random_A, random_C=random_C)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=test_data_str)

encode, decode = get_encode_decode(meta_path)
result_dict = {'prompt_set_num':[], 'prompt_num':[], 'num_train_samples':[], 'test_accuracy':[], 'num_ar': [], 'num_add': [], 'mean_accuracy': [], 'this_mean_accuracy': [], 'at_least_one_correct': []}
result_dir = get_results_dir(config)
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
    elif select == 'fewshot':
        # ckpt_ar3000_add2000_final
        if num_ar == 0:
            num_ar = '*'
        if num_add == 0:
            num_add = '*'
            # ckpt_ar3000_add2000_final
        ckpt_list = glob.glob(out_dir+f'/ckpt_ar{num_ar}_add{num_add}_*final.pt')
    
    if not select == 'fewshot':
        ckpt_list.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
else:
    ckpt_list = [os.path.join(out_dir, ckpt_path_name)]

print(ckpt_list)

def get_num_prompt_and_set(x):
    num_prompt = int(x.split('_')[-2].split('.')[0])
    num_set = int(x.split('_')[-1].split('.')[0])
    return num_prompt, num_set
        
for ckpt_path_name in ckpt_list:
    # load model
    print(f'loading model from: {ckpt_path_name}')
    num_train = int(''.join(x for x in ckpt_path_name if x.isdigit())) if plot_sample_acc else None
    
    if select == 'samplenum':
        num_add = int(''.join(x for x in ckpt_path_name.split('/')[-1].split('_')[-2].split('_')[0] if x.isdigit())) if plot_sample_acc else None
        num_ar = None
    elif select == 'fewshot':
        num_add = int(''.join(x for x in ckpt_path_name.split('/')[-1].split('add')[1].split('_')[0] if x.isdigit())) if plot_sample_acc else None
        num_ar = int(''.join(x for x in ckpt_path_name.split('/')[-1].split('ar')[1].split('_')[0] if x.isdigit())) if plot_sample_acc else None

    checkpoint = torch.load(ckpt_path_name, map_location=device)
    model = load_trained_model(config, checkpoint)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
    
    # process the start_file
    if multiple_set_per_prompt:
        start_pattern = '_'.join(start.split('FILE:')[1].split('_')[:-2])
        start_list_all = glob.glob(start_pattern+'_*_*.txt')
        total_prompt = len(set([get_num_prompt_and_set(x)[0] for x in start_list_all]))

        start_list_list = []

        for i in range(1, total_prompt+1):
            start_list = [x for x in start_list_all if get_num_prompt_and_set(x)[0] == i]
            start_list.sort(key=lambda x : get_num_prompt_and_set(x))   
            start_list_list.append(start_list)

    else:
        start_pattern = '_'.join(start.split('FILE:')[1].split('_')[:-1])
        start_list2 = glob.glob(start_pattern+'_?.txt')
        if len(start_list2) == 0:
            start_list2 = [start.split('FILE:')[-1]]
        else:
            start_list2.sort(key=lambda x : int(x.split('_')[-1].split('.')[0]))
        total_prompt = len(start_list2)
    # start_list.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
    print('evaluating prompts: ', start_list_list if multiple_set_per_prompt else start_list2)


    prompt_acc_dict = {f'prompt_{i}_accuracy': 0 for i in range(1, total_prompt+1)}
    acc_list_total = []

    # big for loop to go through each prompt examples (ex. prompt1, prompt2, etc.)
    for prompt_num in range(1, total_prompt+1):
        if multiple_set_per_prompt:
            start_list = start_list_list[prompt_num-1]
        else:
            start_list = [start_list2[prompt_num-1]]

        prompt_i_acc_list = []
        # for loop to go through each subset within prompt (ex. prompt1_set1, prompt1_set2, etc.)
        for i, start_file in enumerate(start_list):    
            config['start'] = 'FILE:' + start_file
            # evaluate addition performance
            test_accuracy, accuracy_dict, acc_list = evaluate_addition_fewshot_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                                                                     reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                                                     binary=binary, fewshot=fewshot, data_type=data_type, operator=operator, data_format=data_format)
            acc_list_total.append(acc_list)

            this_mean_accuracy = np.mean(acc_list) * 100
            if plot_sample_acc and wandb_log:
                wandb_dict={"num_ar": num_ar, "num_add": num_add, "this_mean_accuracy": this_mean_accuracy}
                result_dict['prompt_set_num'].append(prompt_num); result_dict['prompt_num'].append(i+1)
                result_dict['num_ar'].append(num_ar); result_dict['num_add'].append(num_add); result_dict['num_train_samples'].append(None)
                result_dict['this_mean_accuracy'].append(this_mean_accuracy); result_dict['test_accuracy'].append(None); result_dict['mean_accuracy'].append(None); result_dict['at_least_one_correct'].append(None)

            elif wandb_log:
                wandb_dict = {"this_mean_accuracy": this_mean_accuracy}
                result_dict['prompt_set_num'].append(prompt_num); result_dict['prompt_num'].append(i+1)
                result_dict['num_ar'].append(None); result_dict['num_add'].append(None); result_dict['num_train_samples'].append(None)
                result_dict['this_mean_accuracy'].append(this_mean_accuracy); result_dict['test_accuracy'].append(None); result_dict['mean_accuracy'].append(None); result_dict['at_least_one_correct'].append(None)
            
            if wandb_log:
                wandb_dict.update(accuracy_dict)
                wandb.log(wandb_dict)
            prompt_i_acc_list.append(this_mean_accuracy)
            
            result_df = pd.DataFrame(result_dict)
            result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
        
        prompt_i_mean_acc = np.mean(prompt_i_acc_list)
        prompt_acc_dict[f'prompt_{prompt_num}_accuracy'] = prompt_i_mean_acc


    all_mean_accuracy = np.mean(acc_list_total) * 100
    at_least_one_correct = np.mean(np.sum(np.array(acc_list_total) > 0, axis=0) > 0) * 100

        
    if plot_sample_acc and wandb_log:
        wandb_dict={"num_ar": num_ar, "num_add": num_add, "mean_accuracy": all_mean_accuracy, "at_least_one_correct": at_least_one_correct}
        result_dict['num_ar'].append(num_ar); result_dict['num_add'].append(num_add); result_dict['mean_accuracy'].append(all_mean_accuracy); result_dict['at_least_one_correct'].append(at_least_one_correct)
        result_dict['prompt_set_num'].append(None); result_dict['prompt_num'].append(None)
        result_dict['num_train_samples'].append(None); result_dict['this_mean_accuracy'].append(None); result_dict['test_accuracy'].append(None)
        wandb_dict.update(prompt_acc_dict)
        wandb.log(wandb_dict)    
        

    elif wandb_log:
        wandb_dict={"test_accuracy": all_mean_accuracy}
        wandb_dict.update(accuracy_dict)
        wandb.log(wandb_dict)
        result_dict['prompt_set_num'].append(None); result_dict['prompt_num'].append(None)
        result_dict['num_ar'].append(None); result_dict['num_add'].append(None); result_dict['num_train_samples'].append(None)
        result_dict['this_mean_accuracy'].append(None); result_dict['test_accuracy'].append(None); result_dict['mean_accuracy'].append(all_mean_accuracy); result_dict['at_least_one_correct'].append(at_least_one_correct)

    
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
    
