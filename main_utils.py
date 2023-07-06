import os
import torch
import numpy as np
from tqdm import tqdm
import random
import math
import string
import pickle
import copy
import pandas as pd
import tiktoken

from model import GPTConfig, GPT

def load_trained_model(config, checkpoint=None):
    init_from = config['init_from']
    model_type = config['model_type']

    if model_type == 'gpt2':
        if init_from == 'resume':
            
            # TODO: Previous NanoGPT model that didn't use flash attention (due to dropout=0.2 for previous version not supporting)
            # has 'attn.bias' in the saved checkpoints. This causes error when creating and loading existing checkpoints 
            # set use_flash = False if 'attn.bias' in in any of the keys in checkpoint['model'].keys()

            if any('attn.bias' in key for key in checkpoint['model'].keys()):
                checkpoint['model_args']['use_flash'] = False
            
            # init from a model saved in a specific directory
            gptconf = GPTConfig(**checkpoint['model_args'])

            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    elif model_type == 'gpt2_rpe':
        from model_rpe import GPTConfig_RPE, GPT_RPE
        if init_from == 'resume':
            # init from a model saved in a specific directory
            gptconf = GPTConfig_RPE(**checkpoint['model_args'])
            model = GPT_RPE(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            model = GPT_RPE.from_pretrained(init_from, dict(dropout=0.0))
    

    elif model_type == 'lstm':
        from model_lstm import LSTMConfig, RNNModule
        if init_from == 'resume':
            # init from a model saved in a specific directory
            gptconf = LSTMConfig(**checkpoint['model_args'])
            model = RNNModule(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            pass
    
    return model

def get_results_dir(config):
    results_dir = config['out_dir']+'/'
    # results_dir += config['dataset']+'_'
    if config['exp_name'] == 'default_exp_name':
        config['exp_name'] = config['wandb_run_name']
        
    results_dir += config['exp_name']

    if os.path.exists(results_dir):
        print(f"WARNING: results directory {results_dir} already exists, overwriting...")
        id = 1
        while os.path.exists(results_dir+'_'+str(id)):
            id += 1
        results_dir += '_'+str(id)
    
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def is_number(s):
    # handle "xey" case (e.g. 1.2e-3) - we do not use this notation in our dataset
    if 'e' in s:
        return False
    elif 'E' in s:
        return False
    elif 'inf' in s or "INF" in s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_num_digits(a: str):
    if a == '':
        return 0
    else:
        if '.' in a: # if a contains a decimal point
            return len(a) - 1
        else:
            return len(str(int(a)))


def convert_to_binary(num):
    return bin(num).replace("0b", "")


def convert_to_decimal(num):
    return int(num, 2)


def numCarryOps(a, b, binary=False):
    def digitSum(n):
        return sum(map(int,str(n)))
    if b == '':
        return 0
    
    if not binary:
        a,b=int(a),int(b)        
        # assert(a >= 0); assert(b >= 0);
        return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)
    else:
        c = int(a,2) + int(b,2)
        return int((digitSum(a) + digitSum(b) - digitSum(convert_to_binary(c))) )


def get_encode_decode(meta_path=None, tokenizer='char'):
    import pickle, tiktoken
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if meta_path and tokenizer == 'char':
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif tokenizer:
        print(f"Trying to load tiktoken's openAI {tokenizer} tokenizer")
        enc = tiktoken.get_encoding(f"{tokenizer}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


def get_batch(data, batch_size, block_size, device):
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y


def evaluate_text(config, model, data, ctx):
    ### https://huggingface.co/docs/transformers/perplexity
    model.eval()
    device = config['device']
    batch_size = config['batch_size']
    block_size = model.config.block_size
    stride = block_size // 2
    model_type = config['model_type'] if 'model_type' in config.keys() else 'gpt2'

    seq_len = len(data)
    
    nlls = []
    prev_end_loc = 0
    begin_loc = 0

    while True:
        if begin_loc + block_size > seq_len -1:
                break
        X = []
        Y = []
        for count in range(batch_size):
            if begin_loc + block_size > seq_len -1:
                break
            end_loc = min(begin_loc + block_size, seq_len-1)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            X.append(torch.from_numpy((data[begin_loc:end_loc]).astype(np.int64)).to(device))
            Y.append(torch.from_numpy((data[begin_loc+1:end_loc+1]).astype(np.int64)).to(device))

            begin_loc += stride
            prev_end_loc = end_loc

        X = torch.stack(X)
        Y = torch.stack(Y)

        with torch.no_grad():
            with ctx:
                logits, loss = model(X, Y)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = loss * trg_len

        nlls.append(neg_log_likelihood)
        # print(neg_log_likelihood)

        
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    # ppl = torch.stack(nlls).sum() / end_loc
    print(f'perplexity: {ppl}')
    model.train()

    return ppl


def remove_zero_pad(a: str):
    assert(all([i=='0' for i in a[::2]]))
    return a[1::2]


def reverse_string(a: str) -> str:
    a = str(a)
    return a[::-1]
    

def get_abc_new(abc: str, zero_pad=False, reverse_ab=False, binary=False, few_shot=False, algo_reason=False):
    if few_shot:
        if algo_reason:
            abc = abc.split('Target')[-2]
        abc = abc.strip().split('\n')[-1]
    if 'sin(' in abc:
        operation = 'sin'
    elif 'sqrt(' in abc:
        operation = 'sqrt'
    elif '+' in abc:
        operation = '+'
    elif '-' in abc:
        operation = '-'
    elif '*' in abc:
        operation = '*'

    else:
        print(f'operation not found, abc: {abc}')

    if operation in ['+', '-', '*']:
        [a,b] = abc.split(operation)

    elif operation in ['sin', 'sqrt']:
        if 'Input:' in abc:
            a = abc.split('Input:\n')[-1].split('\nTarget')[0]
        else:
            # a, _ = abc.strip().split('=')
            a = abc.strip().split('=')[0]
        a = a.replace(operation, '').replace('(', '').replace(')', '')
        b = ''

    if a[0] == '$':
        a = a[1:]
    if a.startswith('Input:\n'):
        a = a.split('Input:\n')[-1]
    if 'Target' in b:
        b = b.split('\nTarget')[0]

    b = b.split('=')[0]
    
    a = a.replace(' ', '')
    b = b.replace(' ', '')

    if binary:
        c = int(a,2) + int(b,2)
        return a, b, int(convert_to_binary(c))
    
    if zero_pad:
        a, b = remove_zero_pad(a), remove_zero_pad(b)
    
    if reverse_ab:
        a, b = reverse_string(a), reverse_string(b)
    
    if operation == '+': c = int(a) + int(b)
    elif operation == '-': c = int(a) - int(b)
    elif operation == '*': c = int(a) * int(b)
    elif operation == 'sin': 
        if algo_reason:
            c = get_algo_reasoning_str(float(a), operator='sin')
            c = c.split('</scratch>\n')[1].rstrip()
            c = float(c)
        else:
            c = math.floor(math.sin(float(a)) * 10000) / 10000
    elif operation == 'sqrt': 
        if algo_reason:
            c = get_algo_reasoning_str(float(a), operator='sqrt')
            c = c.split('</scratch>\n')[1].rstrip()
            c = float(c)
        else:
            c = math.floor(math.sqrt(float(a)) * 10000) / 10000
    
    if '\n' in b: b = b[:-1]

    return a,b,c,operation


def get_error_metric(y, y_hat, metric_type='accuracy', eps=0, list_not_num=[], list_outlier_num=[]):
    # if type(y) not in [float, int] or type(y_hat) not in [float, int]:
        #  handle cases ex. y_hat is sin(-0.3241)=-0.318(
        # return np.NaN # TODO: what will be the best way to handle this?
        # if y_hat is not a number, then num_not_num += 1
        # else include y_hat in computation
        # also report num_not_num/total_num * 100
    if type(y_hat) not in [float, int]:
        # if y_hat contains non-numbers or '.', '-', delete that character from the string
        list_not_num.append(copy.deepcopy(y_hat))
        # TODO: code to fix y_hat and use it. Might want to uncomment later.
        # y_hat = str(y_hat)
        # y_hat_new = ''
        # for i in range(len(y_hat)):
        #     if y_hat[i] in ['0','1','2','3','4','5','6','7','8','9','.','-']:
        #         y_hat_new += y_hat[i]
        # y_hat = float(y_hat_new)
        # y = float(y)
        print(f"Skipping y_hat={y_hat}")
        error = np.NaN

    elif metric_type == 'accuracy':
        if eps == 0:
            error = np.abs(y == y_hat)
        else:
            error = np.abs(y - y_hat) < eps

    elif metric_type == 'mse':
        error = (y - y_hat)**2
        if error > 10:
            print(f"Skipping y_hat={y_hat}")
            list_outlier_num.append(copy.deepcopy(y_hat))
            error = np.NaN

    elif metric_type == 'normalized_mse':
        if y == 0:
            error = ((y - y_hat) / (y+1e-17))**2
        else:
            error = ((y - y_hat) / y)**2
        if error > 10e4:
            print(f"Skipping y_hat={y_hat}")
            list_outlier_num.append(copy.deepcopy(y_hat))
            error = np.NaN

    elif metric_type == 'digit_wise_difference':
        error = np.sum(np.abs(y - y_hat) > eps) / get_num_digits(str(y))

    elif metric_type == 'incorrect_digit_count':
        #  count the number of digits that are incorrect
        count = 0
        y, y_hat = str(y), str(y_hat)
        y, y_hat = y.replace('.', ''), y_hat.replace('.', '')
        y, y_hat = y.replace('-', ''), y_hat.replace('-', '')
        len1 = len(y)
        len2 = len(y_hat)
        for i in range(max(len1, len2)):
            if i > len1-1:
                y_i = 0
                y_hat_i = int(y_hat[i])
            elif i > len2-1:
                y_i = int(y[i])
                y_hat_i = 0
            else:
                y_i = int(y[i])
                y_hat_i = int(y_hat[i])
            if y_i != y_hat_i:
                count += 1

        error = count

    return error, list_not_num, list_outlier_num


def evaluate_addition_new(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False, algo_reason=False, binary=False, fewshot=False):
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+1
    if algo_reason:
        max_new_tokens = 80 if 'simple' in config['dataset'] else 320 # TODO:
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            # start = f.read()
            lines = [line.rstrip() for line in f]
            if algo_reason:
                lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines = lines2
    else:
        lines = start.splitlines()
        if algo_reason:
            lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
            lines = lines2
    correct = 0
    total = len(lines)
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    # digit_dictionary['f']

    x = range(total) if verbose else tqdm(range(total))
    for i in x:
    # for i in tqdm(range(total)):
        line = lines[i]
        line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        len_x = len(x[0])
        a,b,c,op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, few_shot=fewshot, algo_reason=algo_reason)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b, binary=binary)
        # print(start)

        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome = decode(y[0].tolist())
                # if not 'test' in start:
                #     print(c_hat)
                # c_hat = c_hat.split('+')[-1].split('=')[-1]
                c_hat = outcome[len_x:]

                # consider the case where c_hat contains non-digit number ('+','=','\n')
                
                # if '$' == c_hat[-1]:
                #     c_hat = c_hat[:-1]
                
                if '$' == line[0]: # handle $ prompt $
                    c_hat = c_hat.split('$')[0]
                else:
                    if '\n' == c_hat[-1]: # handle cases where it ends with '\n'
                        c_hat = c_hat[:-1]
                
                c_hat2 = c_hat
                if zero_pad:
                    c_hat2 = remove_zero_pad(c_hat)
                
                if algo_reason:
                    if '</scratch>\n' in c_hat: 
                        c_hat2 = c_hat.split('</scratch>\n')[1].split('\n')[0]
                        c_hat2 = c_hat2.replace(' ','')
                    if 'simple' in config['dataset'] and '.\n' in c_hat:
                        c_hat2 = c_hat2.split('.\n')[1]
                        c_hat2 = c_hat2.split('\n')[0]
                else: # plain addition
                    c_hat2 = c_hat2.split('\n')[0]
                
                if reverse_c:
                    c_hat2 = reverse_string(c_hat2)

                if all(elem in "0123456789" for elem in c_hat2) and c_hat != '' and c_hat2 != '':
                    c_hat2 = int(c_hat2)
                else:
                    c = str(c)

                if c == c_hat2:
                    correct+=1
                    carry_dictionary[f'carry{num_carry}_correct']+=1
                    # if verbose:
                    #     print('outputs(o): ', outcome)
                    #     print(f'correct: {a}+{b}={c}')
                else:
                    if verbose:
                        pass
                        print('outputs(x): ', outcome)
                        print(f'wrong  : {a}{op}{b}={c_hat2}')
                        print(f'correct: {a}{op}{b}={c}')


                carry_dictionary[f'carry{num_carry}_total']+=1
    
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
        for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    model.train()
    
    return accuracy, accuracy_dictionary


# making a function to batch evaluate addition
# same as evaluate_addition_new but with batch-wise generation
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False,
                            algo_reason=False, binary=False, fewshot=False, data_type='binary', operator='+', data_format='plain', verbose_correct=False, analyze=False):
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    simple= config['simple'] if 'simple' in config.keys() else False
    add_space = config['add_space'] if 'add_space' in config.keys() else False

    if algo_reason:
        max_new_tokens = 80 if ('simple' in config['dataset'] or simple) else 320 # TODO:
    elif add_space:
        max_new_tokens = 2 * num_digit + 3
    
    if 'multi_digit' in config.keys() and config['multi_digit']:
        if algo_reason and not simple: # TODO: Rough estimate
            if num_digit == 1 : max_new_tokens = 160
            elif num_digit == 2 : max_new_tokens = 220
            elif num_digit == 3: max_new_tokens = 290
            elif num_digit == 4: max_new_tokens = 370
            elif num_digit == 5: max_new_tokens = 450
            elif num_digit == 6: max_new_tokens = 540
            elif num_digit == 7: max_new_tokens = 630
            elif num_digit == 8: max_new_tokens = 800
            else: max_new_tokens = 1000 # TODO:
        if algo_reason and simple:
            max_new_tokens = 20 + 15 * num_digit

    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    eps = config['eps'] if 'eps' in config.keys() else 0
    
    random_A= config['random_A'] if 'random_A' in config.keys() else False
    random_C= config['random_C'] if 'random_C' in config.keys() else False

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

    if data_type=='text':
        test_data_file = start[5:]
        print(f"Evaluating Addition using test data file: {test_data_file}")
        # we know test examples are test.txt
        test_data_list = get_data_list(test_data_file, operator=operator)
        test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=True, add_space=add_space, simple=simple, random_A=random_A, random_C=random_C)
        if algo_reason:
            lines = [x.strip() + "\nTarget:\n" for x in test_data_str.split("Target:")]
            lines = lines[:-1]
        else:
            lines = test_data_str.split('\n')[:-1]
    else:
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                # start = f.read()
                lines = [line.rstrip() for line in f]
                if algo_reason:
                    lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                    lines = lines2
        else:
            lines = start.splitlines()
            if algo_reason:
                lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines = lines2
    correct = 0
    total = len(lines)
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    # digit_dictionary['f']
    if analyze:
        # analyze various metric
        error_dict = {'y':[], 'y_hat':[], 'accuracy_eps0':[], 'accuracy_eps5e-4':[], 'accuracy_eps5e-3':[], 'mse':[], 'normalized_mse':[], 'digit_wise_difference':[], 'incorrect_digit_count':[]}

    line_idxs = range(total) if verbose else tqdm(range(total))
    prompt_dict = {}
    for line_idx in line_idxs:
        line = lines[line_idx]
        line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        len_x = len(x[0])
        a,b,c,op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, few_shot=fewshot, algo_reason=algo_reason)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b, binary=binary)
        prompt_length = len(start_ids)
        # NOTE: prompt_length != len(line) if we're not using character level tokenization
        input_tuple = (x, len(line), line[0], a, b, c, a_d, b_d, num_carry)
        if prompt_length in prompt_dict.keys():
            prompt_dict[prompt_length].append(input_tuple)
        else:
            prompt_dict[prompt_length] = [input_tuple]

    # construct batches of prompts now
    batch_list = []
    list_not_num = []
    list_outlier_num = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)): # range(len(input_tuple_list)//test_batch_size+1):
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])

    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)
        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome_list = [decode(y_i.tolist()) for y_i in y]
                for i, outcome in enumerate(outcome_list):
                    _, len_x, line_start, a, b, c, a_d, b_d, num_carry = batch[i]
                    # if not 'test' in start:
                    #     print(c_hat)
                    # c_hat = c_hat.split('+')[-1].split('=')[-1]
                    c_hat = outcome[len_x:]

                    # consider the case where c_hat contains non-digit number ('+','=','\n')
                    
                    if '$' == line_start: # handle $ prompt $
                        c_hat = c_hat.split('$')[0]
                    else:
                        if '\n' == c_hat[-1]: # handle cases where it ends with '\n'
                            c_hat = c_hat[:-1]
                    
                    c_hat2 = c_hat
                    if zero_pad:
                        c_hat2 = remove_zero_pad(c_hat)
                    
                    if algo_reason:
                        if '</scratch>\n' in c_hat: 
                            c_hat2 = c_hat.split('</scratch>\n')[1].split('\n')[0]
                            c_hat2 = c_hat2.replace(' ','')
                        if ('simple' in config['dataset'] or config['simple'] )and '.\n' in c_hat:
                            c_hat2 = c_hat2.split('.\n')[1]
                            c_hat2 = c_hat2.split('\n')[0]
                    else: # plain addition
                        c_hat2 = c_hat2.split('\n')[0]
                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)
                    
                    if add_space:
                        c_hat2 = c_hat2.replace(' ','')

                    if is_number(c_hat2):
                        if '.' in c_hat2:
                            c_hat2 = float(c_hat2)
                        else:
                            c_hat2 = int(c_hat2)
                    else: # c_hat2 is not a number
                        c = str(c)

                    if op in ['+','-','*']:
                        if c == c_hat2:
                            correct+=1
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {a}{op}{b}={c}')
                        else:
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {a}{op}{b}={c_hat2}')
                                print(f'correct: {a}{op}{b}={c}')
                    elif op in ['sin', 'sqrt']:
                        if type(c)!= str and abs(c-c_hat2)<= eps:
                            correct+=1
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {op}({a})={c}')
                        else:
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {op}({a})={c_hat2}')
                                print(f'correct: {op}({a})={c}')

                    carry_dictionary[f'carry{num_carry}_total']+=1

                    metric_types = ['mse', 'normalized_mse', 'digit_wise_difference', 'incorrect_digit_count']
                    if analyze:
                        error_dict['y'].append(c)
                        error_dict['y_hat'].append(c_hat2)
                        for metric_type in metric_types:
                            error, list_not_num, list_outlier_num = get_error_metric(c, c_hat2, metric_type, eps=config['eps'], list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                            error_dict[f'{metric_type}'].append(error)
                        error, _, _ = get_error_metric(c, c_hat2, 'accuracy', eps=0, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps0'].append(error * 100)
                        error, _, _= get_error_metric(c, c_hat2, 'accuracy', eps=5e-4, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-4'].append(error * 100)
                        error, _, _ = get_error_metric(c, c_hat2, 'accuracy', eps=5e-3, list_not_num=list_not_num, list_outlier_num=list_outlier_num)
                        error_dict[f'accuracy_eps5e-3'].append(error * 100)
    
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
        for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    if analyze:
        error_df = pd.DataFrame(error_dict)
        result_dir = config['result_dir'] if 'result_dir' in config.keys() else get_results_dir(config)
        error_df.to_csv(os.path.join(result_dir, 'error_df.csv'), index=False)
        error_mean_dict = {metric_type: np.nanmean(error_dict[f'{metric_type}']) for metric_type in ['accuracy_eps0', 'accuracy_eps5e-4', 'accuracy_eps5e-3', 'mse', 'normalized_mse','digit_wise_difference', 'incorrect_digit_count']}
        error_mean_dict['num_not_num'] = len(list_not_num) / len(metric_types)
        error_mean_dict['num_outlier_num'] = len(list_outlier_num) / len(metric_types)
        error_mean_dict['median_mse'] = error_df.mse.median()
        error_mean_dict['median_normalized_mse'] = error_df.normalized_mse.median()
        accuracy_dictionary.update(error_mean_dict)
        print('skipped since not a number: ', list_not_num)
        print('skipped since outlier number: ', list_outlier_num)
    model.train()

    return accuracy, accuracy_dictionary


# for evaluatin multiple digit addition
def evaluate_addition_multidigit(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False, algo_reason=False, binary=False):
    model.eval()
    device = config['device']
    if algo_reason:
        max_new_tokens = 80 if 'simple' in config['dataset'] else 320 # TODO:
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    digit_accuracy_dictionary={f'digit{i}_accuracy':0 for i in range(1, num_digit+2)}
    
    for digit in range(1, num_digit+2):
        max_new_tokens = digit + 2
        start = f"FILE:data/{config['dataset']}/prompt_{digit}digit_100.txt"
        print(f'evaluating addition from: {start}')
        # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                # start = f.read()
                lines = [line.rstrip() for line in f]
                if algo_reason:
                    lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                    lines = lines2
        else:
            lines = start.splitlines()
            if algo_reason:
                lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines = lines2
        
        
        correct = 0
        total = len(lines)
        carry_dictionary={f'carry{i}_correct':0 for i in range(digit+1)}
        carry_dictionary.update({f'carry{i}_total':0 for i in range(digit+1)})
        # digit_dictionary['f']

        x = range(total) if verbose else tqdm(range(total))
        for i in x:
        # for i in tqdm(range(total)):
            line = lines[i]
            line.strip('\n')
            start_ids = encode(line)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            len_x = len(x[0])
            a,b,c,op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, algo_reason=algo_reason)
            a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b, binary=binary)
            # print(start)

            # run generation
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    outcome = decode(y[0].tolist())
                    # if not 'test' in start:
                    #     print(c_hat)
                    # c_hat = c_hat.split('+')[-1].split('=')[-1]
                    c_hat = outcome[len_x:]

                    # consider the case where c_hat contains non-digit number ('+','=','\n')

                    if '$' == line[0]: # handle $ prompt $
                        c_hat = c_hat.split('$')[0]
                    else:
                        if '\n' in c_hat: # handle cases where it ends with '\n'
                            c_hat = c_hat.split('\n')[0]
                    
                    c_hat2 = c_hat
                    if zero_pad:
                        c_hat2 = remove_zero_pad(c_hat)
                    
                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)
                    
                    if algo_reason:
                        if '</scratch>\n' in c_hat: 
                            c_hat2 = c_hat.split('</scratch>\n')[1].split('\n')[0]
                            c_hat2 = c_hat2.replace(' ','')

                    if all(elem in "0123456789" for elem in c_hat2) and c_hat != '' and c_hat2 != '':
                        c_hat2 = int(c_hat2)
                    else:
                        c = str(c)

                    if c == c_hat2:
                        correct+=1
                        carry_dictionary[f'carry{num_carry}_correct']+=1
                        # if verbose:
                        #     print('outputs(o): ', outcome)
                        #     print(f'correct: {a}+{b}={c}')
                    else:
                        if verbose:
                            pass
                            print('outputs(x): ', outcome)
                            print(f'wrong  : {a}{op}{b}={c_hat2}')
                            print(f'correct: {a}{op}{b}={c}')

                    carry_dictionary[f'carry{num_carry}_total']+=1
        
        accuracy = correct/total*100
        print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
        accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
            for i in range(digit+1)}
        print(accuracy_dictionary)
        digit_accuracy_dictionary[f'digit{digit}_accuracy'] = accuracy
        
    model.train()
    
    return digit_accuracy_dictionary
    

def evaluate_addition_fewshot_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False, 
                                    algo_reason=False, binary=False, fewshot=False, data_type='binary', operator='+', data_format='plain', verbose_correct=False, analyze=False):
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    prompt_dir = config['prompt_dir'] if 'prompt_dir' in config.keys() else None
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    if operator in ['+','-','*']:
        max_new_tokens = max(max_new_tokens, num_digit+2)
    elif operator in ['sin', 'sqrt']:
        max_new_tokens = max(max_new_tokens, 8)
    if algo_reason:
        max_new_tokens = 80 if 'simple' in config['dataset'] else 320 # TODO:
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200
    eps = config['eps'] if 'eps' in config.keys() else 0
    add_space = config['add_space'] if 'add_space' in config.keys() else False
    simple=simple if 'simple' in config.keys() else False
    random_A=random_A if 'random_A' in config.keys() else False
    random_C=random_C if 'random_C' in config.keys() else False

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

    if data_type=='text':
        test_data_file = start[5:]
        print(f"Evaluating Addition using test data file: {test_data_file}")
        # we know test examples are test.txt
        test_data_list = get_data_list(test_data_file, operator=operator)
        test_data_str = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=True, fewshot=fewshot, prompt=prompt_dir, simple=simple, random_A=random_A, random_C=random_C)
        if fewshot:
            data = test_data_str
            lines = data.split('\n\n')[:-1]
        if algo_reason:
            # lines = [x.strip() + "\nTarget:\n" for x in test_data_str.split("Target:")]
            lines2 = [line+'\n' for line in lines]
            lines = lines2
        if not fewshot:
            lines = test_data_str.split('\n')[:-1]
    else:
        # encode the beginning of the prompt
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                # start = f.read()
                if fewshot:
                    data = f.read()
                    lines = data.split('\n\n')[:-1]           
                if algo_reason:
                    # lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                    lines2 = [line+'\n' for line in lines]
                    lines = lines2
        else:
            lines = start.splitlines()
            if algo_reason:
                lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines = lines2
    correct = 0
    total = len(lines)
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    # digit_dictionary['f']

    acc_list = []

    line_idxs = range(total) if verbose else tqdm(range(total))
    prompt_dict = {}
    for line_idx in line_idxs:
        line = lines[line_idx]
        line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        len_x = len(x[0])
        a,b,c,op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, few_shot=fewshot, algo_reason=algo_reason)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b, binary=binary)
        prompt_length = len(start_ids)
        # NOTE: prompt_length != len(line) if we're not using character level tokenization
        input_tuple = (x, len(line), line[0], a, b, c, a_d, b_d, num_carry)
        if prompt_length in prompt_dict.keys():
            prompt_dict[prompt_length].append(input_tuple)
        else:
            prompt_dict[prompt_length] = [input_tuple]

    # construct batches of prompts now
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)): # range(len(input_tuple_list)//test_batch_size+1):
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])

    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)
        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome_list = [decode(y_i.tolist()) for y_i in y]
                for i, outcome in enumerate(outcome_list):
                    _, len_x, line_start, a, b, c, a_d, b_d, num_carry = batch[i]
                    # if not 'test' in start:
                    #     print(c_hat)
                    # c_hat = c_hat.split('+')[-1].split('=')[-1]
                    c_hat = outcome[len_x:]

                    # consider the case where c_hat contains non-digit number ('+','=','\n')
                    
                    # if '$' == c_hat[-1]:
                    #     c_hat = c_hat[:-1]
                    
                    if '$' == line_start: # handle $ prompt $
                        c_hat = c_hat.split('$')[0]
                    else:
                        if '\n' == c_hat[-1]: # handle cases where it ends with '\n'
                            c_hat = c_hat[:-1]
                    
                    c_hat2 = c_hat
                    if zero_pad:
                        c_hat2 = remove_zero_pad(c_hat)
                    
                    if algo_reason:
                        if '</scratch>\n' in c_hat: 
                            c_hat2 = c_hat.split('</scratch>\n')[1].split('\n')[0]
                            c_hat2 = c_hat2.replace(' ','')
                        if ('simple' in config['dataset'] or config['simple'] )and '.\n' in c_hat:
                            c_hat2 = c_hat2.split('.\n')[1]
                            c_hat2 = c_hat2.split('\n')[0]
                    else: # plain addition
                        c_hat2 = c_hat2.split('\n')[0]

                    if reverse_c:
                        c_hat2 = reverse_string(c_hat2)

                    if add_space:
                        c_hat2 = c_hat2.replace(' ','')
                    
                    if is_number(c_hat2):
                        if '.' in c_hat2:
                            c_hat2 = float(c_hat2)
                        else:
                            c_hat2 = int(c_hat2)
                    else: # c_hat2 is not a number
                        c = str(c)

                    if op in ['+','-','*']:
                        if c == c_hat2:
                            correct+=1
                            acc_list.append(1)
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {a}{op}{b}={c}')
                        else:
                            acc_list.append(0)
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {a}{op}{b}={c_hat2}')
                                print(f'correct: {a}{op}{b}={c}')
                    elif op in ['sin', 'sqrt']:
                        if type(c)!= str and abs(c-c_hat2)<= eps:
                            correct+=1
                            acc_list.append(1)
                            carry_dictionary[f'carry{num_carry}_correct']+=1
                            if verbose_correct:
                                print('outputs(o): ', outcome)
                                print(f'correct: {op}({a})={c}')
                        else:
                            acc_list.append(0)
                            if verbose:
                                print('outputs(x): ', outcome)
                                print(f'wrong  : {op}({a})={c_hat2}')
                                print(f'correct: {op}({a})={c}')
                    carry_dictionary[f'carry{num_carry}_total']+=1
    
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
        for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    model.train()

    return accuracy, accuracy_dictionary, acc_list


# function to set seed for all random number generators
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # to make sure GPU runs are deterministic even if they are slower set this to True
    torch.backends.cudnn.deterministic = False
    # warning: this causes the code to vary across runs
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))


# adding functions to streamline data loading/generation
# get data from .txt file -> outputs list of tuples (x1, x2, y, operator) or (x, y, operator)
def get_data_list(filename=None, operator='+', delim=None):
    import re
    data_list = []
    if filename: # read data from file
        if operator in ['text']:
            with open(filename, 'r') as f:
                data = f.read()
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # if first char is $, assume it's a delimiter
                if line[0] == '$':
                    delim = '$'
                if delim:
                    # remove delim from line
                    line = line.replace(delim, '')
                # x1, x2 = line.strip().split(operator)
                if operator in ['+', '-', '*']:
                    x1, x2 = re.split(r'[+\-\*]', line.strip())
                    x2, y = x2.split("=")
                    if operator == '+':
                        y2 = int(x1) + int(x2)
                    elif operator == '-':
                        y2 = int(x1) - int(x2)
                    elif operator == '*':
                        y2 = int(x1) * int(x2)
                    
                    data_list.append((int(x1), int(x2), int(y2), operator))

                elif operator in ['sin', 'sqrt']:
                    x = line.strip().split('=')[0]
                    x = x.replace(operator, '').replace('(', '').replace(')', '')
                    # x = re.findall(r'\d+', x)
                    # x = '.'.join(x)
                    # y = line.strip().split('=')[1]
                    if operator == 'sin':
                        y = math.sin(float(x))
                    elif operator == 'sqrt':
                        y = math.sqrt(float(x))
                    y = math.floor(y * 10000) / 10000

                    data_list.append((float(x), float(y), operator))


    else: # generate random data
        if operator in ['text']:
            # TODO: For now for creating validation dataset, we just use the last 10% of the shakespeare dataset
            with open('data/shakespeare/input.txt', 'r') as f:
                data = f.read()
                n_text = len(data)
                data = data[int(n_text*0.9):]
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            for _ in range(1000):
                if operator in ['+', '-', '*']:
                    x1, x2 = random.randint(0, 999), random.randint(0, 999)
                    if operator == '+':
                        y = x1 + x2
                    elif operator == '-':
                        y = x1 - x2
                    elif operator == '*':
                        y = x1 * x2
                    data_list.append((int(x1), int(x2), int(y), operator))
                
                elif operator in ['sin', 'sqrt']:
                    if operator == 'sin':
                        x = random.uniform(-math.pi/2, math.pi/2)
                        x = math.floor(x * 10000) / 10000
                        y = math.sin(x)
                    elif operator == 'sqrt':
                        x = random.uniform(0, 10)
                        x = math.floor(x * 10000) / 10000
                        y = math.sqrt(x)
                        
                    y = math.floor(y * 10000) / 10000

                    data_list.append((float(x), float(y), operator))

    return data_list


def list_to_string(a):
    a = str(a)
    return a.replace(' ', '')


def truncate_to_n_digit(x, n=4):
    return math.floor(x * (10 ** n)) / (10 ** n)


def get_algo_reasoning_str(x, y=0, operator='+', train=True, n=4, simple=False, random_A=False, random_C=False):
    if operator in ['+', '-', '*']:
        x, y = str(x), str(y)

        len_x, len_y = len(x), len(y)
        list_x, list_y = [int(digit) for digit in str(x)], [int(digit) for digit in str(y)]
        
        output_str = f'Input:\n{x}{operator}{y}\n'
    
    elif operator in ['sin', 'sqrt']:
        x_trunc = truncate_to_n_digit(x, n=n)
        output_str = f'Input:\n{operator}({x_trunc})\n'

    output_str += f'Target:\n'
    
    if not train:
        return output_str

    # if trian mode, construct for algo reasoning for x+y
    if not simple:
        output_str += f'<scratch>\n'
        
        if operator in ['+', '-', '*']:
            output_str += f'{list_to_string(list_x)} has {len_x} digits.\n'
            output_str += f'{list_to_string(list_y)} has {len_y} digits.\n'

    # AR for addition
    if operator == '+': 
        if simple:
            C=0
            A=[]
            for i in range(max(len_x, len_y)):
                a = list_x[-1] if i < len_x else 0
                b = list_y[-1] if i < len_y else 0
                c = a + b + C

                if random_A:
                    random_A = random.randint(0, 9)
                else:
                    random_A = c % 10
                if random_C:
                    random_C = random.randint(0, 1)
                else:
                    random_C = c // 10
                
                output_str += f'A->{random_A} , C->{random_C}\n'

                A.insert(0, c % 10)
                C = c // 10

                list_x = list_x[:-1]
                list_y = list_y[:-1]

            # output_str += f'{list_to_string(list_x)} + {list_to_string(list_y)} , A={list_to_string(A)} C={C} , END\n</scratch>\n'
            if C == 1:
                A.insert(0, 1)
            output_str = output_str[:-1] + f'.\n'
            for a in A:
                output_str += f'{a}'

            return output_str+'\n'
        C=0
        A=[]
        for i in range(max(len_x, len_y)):
            a = list_x[-1] if i < len_x else 0
            b = list_y[-1] if i < len_y else 0
            if operator == '+':
                c = a + b + C
            else:
                raise ValueError(f'Operator {operator} not supported!')
            
            output_str += f'{list_to_string(list_x)} {operator} {list_to_string(list_y)} , A={list_to_string(A)} , C={C} , {a}{operator}{b}+{C}={c} , A->{c%10} , C->{c//10}\n'

            A.insert(0, c % 10)
            C = c // 10

            list_x = list_x[:-1]
            list_y = list_y[:-1]

        output_str += f'{list_to_string(list_x)} + {list_to_string(list_y)} , A={list_to_string(A)} C={C} , END\n</scratch>\n'
        if C == 1:
            A.insert(0, 1)
        for a in A:
            output_str += f'{a} '

        return output_str[:-1]+'\n'
    
    # AR for subtraction
    # ver1. using comparison between x1 and x2
    # elif operator == '-':
    #     x_y = 1
    #     if x >= y:
    #         output_str += f'{x}>={y}\n'
    #     else:
    #         x_y = 0
    #         output_str += f'{x}<{y} : {x}-{y}=-({y}-{x})\n'
    #         x, y = y, x
    #         len_x, len_y = len(x), len(y)
    #         list_x, list_y = list_y, list_x

    #     C=0
    #     A=[]
    #     total_len = max(len_x, len_y)
    #     for i in range(total_len):
    #         a = list_x[-1] if i < len_x else 0
    #         b = list_y[-1] if i < len_y else 0

            
    #         if a - b - abs(C) < 0:
    #             c = a - b + 10 - abs(C)
    #             update = f"{a}{operator}{b}-{abs(C)}+10={c}"
    #         else:
    #             c = a - b - abs(C)
    #             update = f"{a}{operator}{b}-{abs(C)}={c}"

    #         # print(f"i: {i}, a: {a}, b: {b}, C: {C}, a-b-C: {a - b - C}, c: {c}")

    #         output_str += f'{list_to_string(list_x)} {operator} {list_to_string(list_y)} , A={list_to_string(A)} , C={C} , {update} , A->{c} , C->{-1 * ( a - b - abs(C) < 0)}\n'

    #         A.insert(0, c)
    #         C = -1 * ( a - b - abs(C) < 0)

    #         list_x = list_x[:-1]
    #         list_y = list_y[:-1]

    #     output_str += f'{list_to_string(list_x)} - {list_to_string(list_y)} , A={list_to_string(A)} , END\n</scratch>\n'
        
    #     if x_y == 0:
    #         output_str += '-'
    #     for a in A:
    #         output_str += f'{a} '

    #     return output_str[:-1]+'\n'

    # ver2. 
    elif operator == '-':
        if simple:
            C=0
            A=[]
            for i in range(max(len_x, len_y)):
                a = list_x[-1] if i < len_x else 0
                b = list_y[-1] if i < len_y else 0
                c = a - b - abs(C) 
                C = -1 * ( a - b - abs(C) < 0)

                if c < 0 and i < max(len_x, len_y) - 1:
                    trueA = c + 10
                else:
                    trueA = c
                    
                if random_A:
                    randomA = random.randint(0, 9)
                else:
                    randomA = trueA
                if random_C:
                    randomC = -1 * random.randint(0, 1)
                else:
                    randomC = C
                
                output_str += f'A->{randomA} , C->{randomC}\n'

                A.insert(0, trueA)

                list_x = list_x[:-1]
                list_y = list_y[:-1]
            
            a = int(A[0])
            n = len(A) - 1
            b = int(''.join([str(x) for x in A[1:]])) if n > 0 else 0
            result = a * (10 ** n) + b
            output_str += f'{a * (10 ** n)}+{b}={result}.\n'

            if result < 0:
                result_sign = '-'
                    
            else:
                result_sign = ''
            result_str = [int(x) for x in str(abs(result))]

            output_str += result_sign
            for x in result_str:
                output_str += f'{x}'

            return output_str+'\n'
        C=0
        A=[]
        total_len = max(len_x, len_y)
        for i in range(total_len):
            a = list_x[-1] if i < len_x else 0
            b = list_y[-1] if i < len_y else 0

            
            if a - b - abs(C) < 0 and (i < total_len - 1):
                c = a - b + 10 - abs(C)
                update = f"{a}{operator}{b}-{abs(C)}+10={c}"
            else:
                c = a - b - abs(C)
                update = f"{a}{operator}{b}-{abs(C)}={c}"

            # print(f"i: {i}, a: {a}, b: {b}, C: {C}, a-b-C: {a - b - C}, c: {c}")

            output_str += f'{list_to_string(list_x)} {operator} {list_to_string(list_y)} , A={list_to_string(A)} , C={C} , {update} , A->{c} , C->{-1 * ( a - b - abs(C) < 0)}\n'

            A.insert(0, c)
            C = -1 * ( a - b - abs(C) < 0)

            list_x = list_x[:-1]
            list_y = list_y[:-1]

        output_str += f'{list_to_string(list_x)} - {list_to_string(list_y)} , A={list_to_string(A)}\n' # , END\n</scratch>\n'
        
        a = int(A[0])
        n = len(A) - 1
        b = int(''.join([str(x) for x in A[1:]])) if n > 0 else 0
        result = a * (10 ** n) + b
        output_str += f'{a * (10 ** n)}+{b}={result} , END\n</scratch>\n'

        if result < 0:
            result_sign = '-'
            
        else:
            result_sign = ''
        result_str = [int(x) for x in str(abs(result))]

        output_str += result_sign
        for x in result_str:
            output_str += f'{x} '

        return output_str[:-1]+'\n'

    # AR for multiplication
    elif operator == '*':
        C = 0
        # for i in range(max(len_x, len_y)):
        for i in range(len_y):
            # a = list_x[-1] if i < len_x else 0
            b = list_y[-1] if i < len_y else 0
            # c = a + b + C
            A = b * int(x)
            B = A * (10**i)
            C_prev = C
            C += B
            # A, B = num_to_list(A), num_to_list(B)

            output_str += f'{list_to_string(list_x)} * {b} , A={list_to_string([int(digit) for digit in str(A)])} , k={10**i} , B={list_to_string([int(digit) for digit in str(B)])} , C={C_prev}+{B}={C}'

            if not i == len_y - 1:
                output_str += '\n'
            list_y = list_y[:-1]
        
        output_str += ' , END\n</scratch>\n'
        # output_str += f'{list_to_string(list_x)} * {list_to_string(list_y)} , A={list_to_string(A)} C={C} , END\n</scratch>\n'
        
        for a in str(C):
            output_str += f'{a} '

        return output_str[:-1]+'\n'
    
    elif operator == 'sqrt':
        a = x
        x_true = truncate_to_n_digit(math.sqrt(a), n)
        this_x = x_true

        if this_x >= 1:
            this_x = int(this_x)
        else:
            this_x = 0.1
        output_str += f'x_0={this_x}\n'

        for i in range(1, n+1):
            x_i =this_x

            this_x = 0.5 * (this_x + a / this_x)
            this_x = truncate_to_n_digit(this_x, n)

            output_str += f'x_{i}: 1/2*({x_i}+{a}/{x_i})={this_x}, x_{i}={this_x}'

            if not i == n:
                output_str += '\n'

        output_str += ' , END\n</scratch>\n'
        
        output_str += f'{this_x}\n'

        return output_str[:-1]+'\n'

    elif operator == 'sin':
        # Ver 1. where we use x_i as symbol (instead of explicitly writing the value)
        # Also, this version uses x^k instead of x*x...*x
        x_true = truncate_to_n_digit(x, 4)
        this_x = x_true

        output_str += f'x_0={this_x}\n'

        for i in range(1, n+1):
            k = 2*i+1

            x_i =this_x

            this_x = this_x + (-1) ** i * (x ** k) / (math.factorial(k))
            this_x = truncate_to_n_digit(this_x, n)

            plus_minus = '+ 1' if i % 2 == 0 else '- 1'

            output_str += f'x_{i}: x_{i-1} {plus_minus}/{k}! * (x^{k}) , x_{i}={this_x}'

            if not i == n:
                output_str += '\n'

        output_str += ' , END\n</scratch>\n'
        
        output_str += f'{this_x}\n'

        return output_str[:-1]+'\n'

        # x_true = truncate_to_n_digit(x, 4)
        # this_x = x_true

        # output_str += f'x_0={this_x}\n'

        # for i in range(1, n+1):
        #     k = 2*i+1

        #     x_i =this_x

        #     this_x = this_x + (-1) ** i * (x ** k) / (math.factorial(k))
        #     this_x = truncate_to_n_digit(this_x, n)

        #     plus_minus = '+ 1' if i % 2 == 0 else '- 1'

        #     x_power_k_str = '*'.join(['x']*k)
        #     output_str += f'x_{i}: {x_i} {plus_minus}/{k}! * ({x_power_k_str}) , x_{i}={this_x}'

        #     if not i == n:
        #         output_str += '\n'

        # output_str += ' , END\n</scratch>\n'
        
        # output_str += f'{this_x}\n'

        # return output_str[:-1]+'\n'

    else:
        raise ValueError(f'Operator {operator} for algorithmic reasoning not supported!')

def add_spaces(s):
    # add space if character is a digit or '=', else don't add space
    s = ''.join([c + ' ' if c.isdigit() or c in ['=', '.', '+', '-', '*', '('] else c for c in s])
    if s[-1] == ' ':
        s = s[:-1]
    s = s.replace(' \n', '\n')

    return s

# creating a script to take in a list of tuples [(x1, x2, y)] and output a string of the form "x1 x2 y\n"
# this will be used to generate the data for our TF model
def generate_data_str(data_list, operator='+', format='plain', train=True, shuffle=True, fewshot=False, prompt=None, add_space=False, simple=False, random_A=False, random_C=False):
    
    if format == 'algo_reasoning' and add_space:
        # TODO: add_space=True will add a space between each numbers, but not yet supported for algo_reasoning
        raise ValueError("add_space=True not supported for algo_reasoning format!")
    
    if shuffle:
        random.shuffle(data_list)
    
    if fewshot:
        with open(prompt, 'r') as f:
            prompt = f.read()

    # for idx, (x1, x2, y) in enumerate(data_list):
    for idx, data_tuple in enumerate(data_list):
        operator = data_tuple[-1]
        if operator in ['+', '-', '*']:   
            x1, x2, y = data_tuple[0], data_tuple[1], data_tuple[2]     
            if train:
            # create training data (x1+x2=y)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}={y}\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}={y}$\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}={str(y)[::-1]}$\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}={str(y)[::-1]}\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            else:
                # create test data (x1+x2=)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str

        elif operator in ['sin', 'sqrt']:
            x, y = data_tuple[0], data_tuple[1]
        # for idx, (x, y) in enumerate(data_list):
            if train:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})={y}\n"
            else:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})=\n"
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
        
        elif operator in ['text']:
            output_str = data_tuple[0]
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str+'\n\n'
            else:
                data_str += output_str+'\n\n'

    return data_str


# create and save meta file for a given vocabulary
def create_meta_file(vocabulary, input_data_str=None, tokenizer='char'):
    operators_str = string.punctuation
    if vocabulary == 'custom_input_data' and input_data_str:
        print(f"Input file {input_data_str[:100]} specified. Reading data from file...")
        data = input_data_str
        print(f"length of dataset in characters: {len(data):,}")
        vocabulary = 'custom_input_data'
    elif vocabulary == 'numbers_only':
        print(f"Creating meta file for numbers only...")
        data = string.digits + operators_str + ' \n'
    elif vocabulary == 'all_ascii_chars':
        print(f"Creating meta file for all reasonable characters...")
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'
    else:
        raise ValueError(f"Vocabulary {vocabulary} not supported!")
    
    if tokenizer == 'char':
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            print(f"data has {len(data_ids):,} tokens")
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
        def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # data_ids = data_encoder(data)
        # print(f"data has {len(data_ids):,} tokens")
        # # convert to np array for efficiency
        # data_ids = np.array(data_ids, dtype=np.uint16)

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        meta_path = f'meta_{vocabulary}.pkl'

    elif tokenizer == 'gpt2':
        print("Ignore all above messages about the meta file!!!")
        print(f"Tokenizer specified as {tokenizer}. Loading it from tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        # karpathy uses enc.encode_ordinary(), but since there is no decode_ordinary(), I'm switching to .encode()
        def data_encoder(s):
            data_ids = enc.encode(s, allowed_special={"<|endoftext|>"}) # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids

        def data_decoder(l):
            return enc.decode(l) # decoder: take a list of integers, output a string


        meta = {
            'vocab_size': enc.n_vocab,
        }
        meta_path = f'meta_pretrained_gpt2_tokenizer.pkl'

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return meta, meta_path, data_encoder, data_decoder



############################################
######## Deprecated Functions ###############
############################################

# @deprecated: not using this anymore
def get_abc(abc: str):
    [a,b] = abc.split('+')
    b = b.split('=')[0]
    # a,b = int(a), int(b)
    c = int(a) + int(b)
    return a,b,c

def evaluate_addition(config, model, ctx, encode, decode, verbose=False, num_digit=3, analyze=True):
    print("Warning: evaluate_addition is deprecated. Use evaluate_addition_new instead.")
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+1
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            # start = f.read()
            lines = [line.rstrip() for line in f]
    else:
        lines = start.splitlines()

    correct = 0
    total = len(lines)
    digit_dictionary={f'({i},{j})_correct':0 for i in range(1,num_digit+1) for j in range(num_digit+1)}
    digit_dictionary.update({f'({i},{j})_total':0 for i in range(1,num_digit+1) for j in range(num_digit+1)})
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    # digit_dictionary['f']

    x = range(total) if verbose else tqdm(range(total))
    for i in x:
    # for i in tqdm(range(total)):
        line = lines[i]
        line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        len_x = len(x[0])
        a,b,c = get_abc(line)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b)
        # print(start)

        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                c_hat = decode(y[0].tolist())
                # if not 'test' in start:
                #     print(c_hat)
                # c_hat = c_hat.split('+')[-1].split('=')[-1]
                c_hat = c_hat[len_x:]

                # consider the case where c_hat contains non-digit number ('+','=','\n')
                if '$' == line[0]: # handle $ prompt $
                    c_hat = c_hat.split('$')[0]
                else:
                    if '\n' == c_hat[-1]: # handle cases where it ends with '\n'
                        c_hat = c_hat[:-1]
                        
                if all(elem in "0123456789" for elem in c_hat):
                    c_hat = int(c_hat)
                else:
                    c = str(c)

                if c == c_hat:
                    correct+=1
                    if analyze:
                        digit_dictionary[f'({a_d},{b_d})_correct']+=1
                        carry_dictionary[f'carry{num_carry}_correct']+=1
                else:
                    if verbose:
                        print(f'{a}+{b}!={c_hat} (={c})')

                if analyze:
                    digit_dictionary[f'({a_d},{b_d})_total']+=1
                    carry_dictionary[f'carry{num_carry}_total']+=1
    
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'({i},{j})': digit_dictionary[f'({i},{j})_correct']/digit_dictionary[f'({i},{j})_total']*100 if digit_dictionary[f'({i},{j})_total']!=0 else np.nan \
        for i in range(1,num_digit+1) for j in range(1,num_digit+1)}
    accuracy_dictionary.update({f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
        for i in range(num_digit+1)})
    print(accuracy_dictionary)
    
    model.train()
    
    return accuracy, accuracy_dictionary


def evaluate_addition_fewshot(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, reverse_ab=False, reverse_c=False, algo_reason=False, binary=False, fewshot=False):
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+1
    if algo_reason:
        max_new_tokens = 80 if 'simple' in config['dataset'] else 320 # TODO:
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 200

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            # start = f.read()
            if fewshot:
                data = f.read()
                lines = data.split('\n\n')[:-1]           
            if algo_reason:
                # lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
                lines2 = [line+'\n' for line in lines]
                lines = lines2
    else:
        lines = start.splitlines()
        if algo_reason:
            lines2 = [lines[2*i]+'\n'+lines[2*i+1]+'\n' for i in range(len(lines)//2)]
            lines = lines2
    
    correct = 0
    total = len(lines)
    carry_dictionary={f'carry{i}_correct':0 for i in range(num_digit+1)}
    carry_dictionary.update({f'carry{i}_total':0 for i in range(num_digit+1)})
    # digit_dictionary['f']

    acc_list = []

    x = range(total) if verbose else tqdm(range(total))
    for i in x:
    # for i in tqdm(range(total)):
        line = lines[i]
        line.strip('\n')
        start_ids = encode(line)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        len_x = len(x[0])
        a,b,c,op = get_abc_new(line, zero_pad=zero_pad, reverse_ab=reverse_ab, binary=binary, few_shot=fewshot, algo_reason=algo_reason)
        a_d, b_d, num_carry = get_num_digits(a), get_num_digits(b), numCarryOps(a,b, binary=binary)
        # print(start)

        # run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outcome = decode(y[0].tolist())
                # if not 'test' in start:
                #     print(c_hat)
                # c_hat = c_hat.split('+')[-1].split('=')[-1]
                c_hat = outcome[len_x:]

                # consider the case where c_hat contains non-digit number ('+','=','\n')
                
                # if '$' == c_hat[-1]:
                #     c_hat = c_hat[:-1]
                
                if '$' == line[0]: # handle $ prompt $
                    c_hat = c_hat.split('$')[0]
                else:
                    if '\n' == c_hat[-1]: # handle cases where it ends with '\n'
                        c_hat = c_hat[:-1]
                
                c_hat2 = c_hat
                if zero_pad:
                    c_hat2 = remove_zero_pad(c_hat)
                
                if reverse_c:
                    c_hat2 = reverse_string(c_hat)
                
                if algo_reason:
                    if '</scratch>\n' in c_hat: 
                        c_hat2 = c_hat.split('</scratch>\n')[-1].split('\n')[0]
                        c_hat2 = c_hat2.replace(' ','')
                    if 'simple' in config['dataset'] and '.\n' in c_hat:
                        c_hat2 = c_hat2.split('.\n')[1]
                        c_hat2 = c_hat2.split('\n')[0]
                    
                else: # plain addition
                    c_hat2 = c_hat2.split('\n')[0]

                if all(elem in "0123456789" for elem in c_hat2) and c_hat != '' and c_hat2 != '':
                    c_hat2 = int(c_hat2)
                else:
                    c = str(c)

                if c == c_hat2:
                    correct+=1
                    carry_dictionary[f'carry{num_carry}_correct']+=1
                    # if verbose:
                    #     print('outputs(o): ', outcome)
                    #     print(f'correct: {a}+{b}={c}')
                    acc_list.append(1)
                else:
                    if verbose:
                        pass
                        print('outputs(x): ', outcome)
                        print(f'wrong  : {a}{op}{b}={c_hat2}')
                        print(f'correct: {a}{op}{b}={c}')
                    acc_list.append(0)

                carry_dictionary[f'carry{num_carry}_total']+=1
    
    accuracy = correct/total*100
    print(f"accuracy of {total} examples: {correct}/{total} ({accuracy}%)")
    accuracy_dictionary = {f'carry{i}': carry_dictionary[f'carry{i}_correct']/carry_dictionary[f'carry{i}_total']*100 if carry_dictionary[f'carry{i}_total']!=0 else np.nan \
        for i in range(num_digit+1)}
    print(accuracy_dictionary)
    
    model.train()
    
    return accuracy, accuracy_dictionary, acc_list


def print_model_output(model, encode, decode, max_new_tokens=50, temperature=0.8, top_k=200, device='cuda', device_type='cuda', ptdtype=torch.float16, start="Twinkle twinkle little star"):
    num_samples = 1
    # encode the beginning of the prompt
    print('\n-----------------------------------------------------------------------------------------------')
    print(f"Prompting model with {start} for {max_new_tokens} tokens")
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('-----------------------------------------------------------------------------------------------\n')
