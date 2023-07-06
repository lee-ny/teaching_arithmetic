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
test_batch_size = 128
eps = 0.0
tokenizer = 'char' # by default, use char level tokenizer. but for pretrained models, use openai tokenizer eg: 'gpt2'

simple=False
random_A=False
random_C=False

multi_digit=False
multi_model=False

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
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=test_data_str, tokenizer=tokenizer)
    meta_vocab_size = meta['vocab_size']

encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)
if multi_digit:
    digit_accuracy_dictionary = {}
    for digit in range(1, num_digit+1):
        digit_accuracy_dictionary[f"digit_{digit}"] = []

if multi_model:
    out_dir = '/'.join(ckpt_path_name.split('/')[:-1])
    result_dir = out_dir + '/eval_result' 
    ckpt_list = glob.glob(os.path.join(out_dir, '*ckpt_*0_final.pt'))
    result_filename = result_dir + '/eval_result.csv'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
else:
    result_dir = ckpt_path_name.split('.')[0]
    ckpt_list = [ckpt_path_name]
    result_filename = result_dir + '_eval_result.csv'
config['result_dir'] = result_dir

print("ckpt_list: ", ckpt_list)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)

    

result_dict = {"model_name": [], "test_accuracy": [] }
for ckpt_path_name in ckpt_list:
    # load model
    print(f'loading: {ckpt_path_name}')

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
    config['max_new_tokens'] =  num_digit+2
    test_accuracy, accuracy_dict = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=verbose, num_digit=num_digit, zero_pad=zero_pad, 
                                                            reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                            binary=binary, fewshot=fewshot, data_type=data_type, operator=operator, data_format=data_format)
    
    result_dict["model_name"].append(ckpt_path_name)
    result_dict["test_accuracy"].append(test_accuracy)
    # result_dict.update(accuracy_dict)

    if multi_digit:
        this_dictionary = {}
        for digit in range(1, num_digit+1):
            config['max_new_tokens'] =  digit+2
            if digit == 1:
                config['start'] = f"FILE:data/multi_digit/test_{digit}digit_100.txt"
            else:
                if algo_reason:
                    config['start'] = f"FILE:data/multi_digit/test_{digit}digit_100.txt"
                else:
                    config['start'] = f"FILE:data/multi_digit/test_{digit}digit_10000.txt"
            digit_accuracy, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=digit, zero_pad=zero_pad, 
                                                reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                binary=binary, data_type=data_type, operator=operator, data_format=data_format)
            digit_accuracy_dictionary[f"digit_{digit}"].append(digit_accuracy)
            this_dictionary[f"digit_{digit}"] = digit_accuracy
        result_dict.update(digit_accuracy_dictionary)

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(result_filename, index=False)

    end=time.time()
    print("Time taken by evaluation_addition(): {}".format(end-start))
        
    if wandb_log:
        wandb_dict={"model_name": ckpt_path_name, "test_accuracy": test_accuracy}
        # wandb_dict.update(accuracy_dict)
        if multi_digit:
            wandb_dict.update(this_dictionary)
        
        wandb.log(wandb_dict)
