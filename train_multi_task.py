"""
This file is used for experiments in jointly training on all five arithmetic tasks. 
"""

import os
import time
import math
import pickle
import pandas as pd
import yaml

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from main_utils import *

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
resume_dir = None
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_entity = 'ssdd'
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
exp_name = 'default_exp_name'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
train_data_path = 'train.bin'
val_data_path = 'val.bin'
multi_digit = False
num_digit = 5
binary = False
# using two data - data1 = text / data2 = addition
train_both = False # use seperate text/add data for train/val (get_batch uses this to sample from two differernt datasets)
data_ratio = 0.2 # ratio of data_path2 compared with data_path1
train_data_path2 = 'train_addition.bin' # only used when train_both = True
val_data_path2 = 'val_addition.bin'
# evaluation
eval_text = False # if True get perplexity using eval_text_data_path
eval_text_data_path = None # directory to text data (.bin file) - ex. 'data/shakespeare_add_ar_mixed/val_text.bin' 
eval_addition = False # if True compute test accuracy of "a+b="
eval_addition_ar = False
start_ar = None
start = None
eval_addition_train = False
start_train = None
reverse_ab = False
reverse_c = False
zero_pad = False
algo_reason = False
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
ckpt_path_name = 'ckpt.pt'
save_final = True
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = None # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
use_flash = True
data_type = 'binary' # 'binary' by default, can be 'text'
operator = '+' # can be '+', '-', '*', 'sin', 'sqrt'
data_shuffle = True
data_format = 'plain' # 'plain' or 'reverse' or 'algo_reasoning'
vocabulary = 'all_ascii_chars' # can be 'all_ascii_chars' or 'numbers_only' or 'custom_input_data'
meta_path_specified = True # use saved meta_file (False if data_type='text')
eps = 0

use_lora = False # use lora (from minLoRA)

# -----------------------------------------------------------------------------
# import ipdb; ipdb.set_trace()
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if min_lr == None:
    min_lr = learning_rate/10

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print('ddp: ', ddp)
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True # cudnn auto-tuner
torch.backends.cudnn.deterministic = False # cudnn auto-tuner
# this is probably overkill but seed everything agian
set_seed(1337 + seed_offset)


device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if data_type == 'binary':
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, train_data_path), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, val_data_path), dtype=np.uint16, mode='r')
    if train_both:
        train_data2 = np.memmap(os.path.join(data_dir, train_data_path2), dtype=np.uint16, mode='r')
        val_data2 = np.memmap(os.path.join(data_dir, val_data_path2), dtype=np.uint16, mode='r')
    if eval_text:
        if eval_text_data_path is None:
            print('eval_text_data_path is None!!! No binary file to evaluate perplexity on.') 
        eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
    # test_data_str = None # test_data for addition testing will be handled with "start"
    meta_path = None
else:
    # check for data_format
    if data_type == 'text':
        if (data_format == 'reverse' and not reverse_c) or (reverse_c and data_format != 'reverse'):
            raise ValueError('reverse_c must be True for data_format == "reverse"')
        elif (data_format == 'algo_reasoning' and not algo_reason) or (algo_reason and data_format != 'algo_reasoning'):
            raise ValueError('algo_reason must be True for data_format == "algo_reasoning"')
    meta_path_specified = False

    data_dir = os.path.join('data', dataset)
    train_data_path = os.path.join(data_dir, train_data_path)
    # val_data = os.path.join(data_dir, val_data_path)
    # TODO: current just hard-coding everything
    sin_data_list = get_data_list('data/sin/train_sin_10000.txt', operator='sin')
    sqrt_data_list = get_data_list('data/sqrt/train_sqrt_10000.txt', operator='sqrt')
    add_data_list = get_data_list('data/bal/train_3digit_10000.txt', operator='+')
    sub_data_list = get_data_list('data/bal/train_3digit_10000.txt', operator='-')
    mul_data_list = get_data_list('data/bal/train_multiplication_3000.txt', operator='*')
    train_data_list = sin_data_list + sqrt_data_list + add_data_list + sub_data_list + mul_data_list
    train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle)

    # train_data_list = get_data_list(train_data_path, operator=operator)
    sin_val_list = get_data_list(operator='sin')
    sqrt_val_list = get_data_list(operator='sqrt')
    add_val_list = get_data_list(operator='+')
    sub_val_list = get_data_list(operator='-')
    mul_val_list = get_data_list(operator='*')
    val_data_list = sin_val_list + sqrt_val_list + add_val_list + sub_val_list + mul_val_list
    val_data_str = generate_data_str(val_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle)

    # train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle)
    meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=train_data_str)
    meta_vocab_size = meta['vocab_size']
    train_data = data_encoder(train_data_str)
    val_data = data_encoder(val_data_str)
    if eval_addition_train and start_train is None:
        # specify the start_train to be oour train data file
        start_train = f"FILE:{train_data_path}"
        

    if train_both:
        # This is for the case where we use two different datasets for training
        # we sample from both with a specified ratio - data_ratio
        # TODO: let's leave this here for now.
        train_data2 = np.memmap(os.path.join(data_dir, train_data_path2), dtype=np.uint16, mode='r')
        val_data2 = np.memmap(os.path.join(data_dir, val_data_path2), dtype=np.uint16, mode='r')
    
    if eval_text:
        # eval_text_data = np.memmap(eval_text_data_path, dtype=np.uint16, mode='r')
        text_data_list = get_data_list(eval_text_data_path, operator='text')
        text_data_str = generate_data_str(text_data_list, operator='text', format=data_format, train=False, shuffle=False)
        eval_text_data = data_encoder(text_data_str)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    if train_both:
        data2 = train_data2 if split == 'train' else val_data2
        batch_size2 = int(batch_size*data_ratio)
        ix = torch.randint(len(data) - block_size, (batch_size-batch_size2,))
        ix2 = torch.randint(len(data2) - block_size, (batch_size2,))
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if train_both:
        x2 = torch.stack([torch.from_numpy((data2[i:i+block_size]).astype(np.int64)) for i in ix2])
        y2 = torch.stack([torch.from_numpy((data2[i+1:i+1+block_size]).astype(np.int64)) for i in ix2])
        x = torch.cat([x,x2])
        y = torch.cat([y,y2])    

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_perplexity = 1e9 # on text data
best_accuracy = -1 # on addition data

if meta_path_specified:
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    else:
        meta_path = None

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_flash=use_flash) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if resume_dir:
        checkpoint = torch.load(resume_dir, map_location=device)
    else:
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, ckpt_path_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    max_iters += iter_num
    best_val_loss = checkpoint['best_val_loss']
    if 'best_perplexity' in checkpoint.keys(): 
        best_perplexity = checkpoint['best_perplexity']
    if 'best_accuracy' in checkpoint.keys():
        best_accuracy = checkpoint['best_accuracy']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if use_lora:
    import minlora
    import inspect
    minlora.add_lora(model, lora_config=lora_config)
    minlora.tie_weights(linear=model.lm_head, embedding=model.transformer.wte)
    # optimizer
    def configure_optimizers_lora(self, weight_decay, learning_rate, betas, device_type):
        # we apply weight decay to all lora params
        optim_groups = [
            # note: .get_lora_params() returns a generator
            # we need to wrap it in a list so we can consume it twice
            {"params": list(minlora.get_lora_params(self)) , "weight_decay": weight_decay},
            # you can also add biases for fine-tuning,
            # but I want to make sure lora alone works
            # {"params": minlora.get_bias_params(self), "weight_decay": 0.0}, # bias params don't get weight decay
        ]

        def parameter_count(optim_groups):
            n = sum(p.numel() for group in optim_groups for p in group["params"])
            if n < 1e6:
                return f"{n/1e3:.1f}k"
            else:
                return f"{n/1e6:.1f}M"

        print(f"optimizing {parameter_count(optim_groups)} parameters")

        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
if use_lora:
    optimizer = configure_optimizers_lora(model, weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
encode, decode = get_encode_decode(meta_path)

result_dict = {'iter': [], 'train_loss': [], 'val_loss': [], 'val_ppl': [], 
               'test_acc': [], 'test_acc_sin': [], 'test_acc_sqrt': [], 'test_acc_sub': [], 'test_acc_mul': [],
               'train_acc': [], 'train_acc_sin': [], 'train_acc_sqrt': [], 'train_acc_sub': [], 'train_acc_mul': [],}
result_dir = get_results_dir(config)
config['result_dir'] = result_dir
with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        if eval_text:
            ppl = evaluate_text(config, model, eval_text_data, ctx)
            print(f"perplexity of evluation text data: {ppl}")

        config['start'] = "FILE:data/bal/test_3digit_10000.txt" if not algo_reason else "FILE:data/bal/test_3digit_100.txt"
        test_accuracy, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="+", data_format=data_format)
        config['start'] = "FILE:data/sin/test_sin_10000.txt" if not algo_reason else "FILE:data/sin/test_sin_100.txt"
        test_accuracy_sin, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="sin", data_format=data_format)
        config['start'] = "FILE:data/sqrt/test_sqrt_10000.txt" if not algo_reason else "FILE:data/sqrt/test_sqrt_100.txt"
        test_accuracy_sqrt, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="sqrt", data_format=data_format)
        config['start'] = "FILE:data/bal/test_3digit_10000.txt" if not algo_reason else "FILE:data/bal/test_3digit_100.txt"
        test_accuracy_sub, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="-", data_format=data_format)
        config['start'] = "FILE:data/bal/test_multiplication_7000.txt" if not algo_reason else "FILE:data/bal/test_multiplication_100.txt"
        test_accuracy_mul, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="*", data_format=data_format)
        if False: #eval_addition_ar:
            config['start'] = start_ar
            test_accuracy_ar, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=True, 
                                                       binary=binary, data_type=data_type, operator=operator, data_format='algo_reasoning')
        # let's evaluate on train dataset only one in a while:
        if iter_num % (10 * eval_interval) == 0  and iter_num > 0:
            config['start'] = 'FILE:data/bal/train_3digit_10000.txt'
            train_accuracy, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="+", data_format=data_format)
            config['start'] = 'FILE:data/sin/train_sin_10000.txt'
            train_accuracy_sin, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="sin", data_format=data_format)
            config['start'] = 'FILE:data/sqrt/train_sqrt_10000.txt'
            train_accuracy_sqrt, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad, 
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="sqrt", data_format=data_format)
            config['start'] = 'FILE:data/bal/train_3digit_10000.txt'
            train_accuracy_sub, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad,   
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="-", data_format=data_format)
            config['start'] = 'FILE:data/bal/train_multiplication_3000.txt'
            train_accuracy_mul, _ = evaluate_addition_batch(config, model, ctx, encode, decode, verbose=True, num_digit=num_digit, zero_pad=zero_pad,   
                                                       reverse_ab=reverse_ab, reverse_c=reverse_c, algo_reason=algo_reason, 
                                                       binary=binary, data_type=data_type, operator="*", data_format=data_format)
        else:
            train_accuracy = None
            train_accuracy_sin = None
            train_accuracy_sqrt = None
            train_accuracy_sub = None
            train_accuracy_mul = None
            

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage,
                "ppl": ppl if eval_text else None, 
                "test/accuracy": test_accuracy, 
                "test/accuracy_sin": test_accuracy_sin,
                "test/accuracy_sqrt": test_accuracy_sqrt,
                "test/accuracy_sub": test_accuracy_sub,
                "test/accuracy_mul": test_accuracy_mul,
                "train/accuracy": train_accuracy,
                "train/accuracy_sin": train_accuracy_sin,
                "train/accuracy_sqrt": train_accuracy_sqrt,
                "train/accuracy_sub": train_accuracy_sub,
                "train/accuracy_mul": train_accuracy_mul,
            }
            wandb.log(wandb_dict)

        result_dict['iter'].append(iter_num)
        result_dict['train_loss'].append(losses['train'].item())
        result_dict['val_loss'].append(losses['val'].item())
        result_dict['val_ppl'].append(ppl.item() if eval_text else None)
        result_dict['test_acc'].append(test_accuracy)
        result_dict['test_acc_sin'].append(test_accuracy_sin)
        result_dict['test_acc_sqrt'].append(test_accuracy_sqrt)
        result_dict['test_acc_sub'].append(test_accuracy_sub)
        result_dict['test_acc_mul'].append(test_accuracy_mul)
        result_dict['train_acc'].append(train_accuracy)
        result_dict['train_acc_sin'].append(train_accuracy_sin)
        result_dict['train_acc_sqrt'].append(train_accuracy_sqrt)
        result_dict['train_acc_sub'].append(train_accuracy_sub)
        result_dict['train_acc_mul'].append(train_accuracy_mul)

        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)
        
        checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'best_perplexity': best_perplexity,
                    'best_accuracy': best_accuracy,
                    'config': config,
                }
        if use_lora:
            checkpoint['lora'] = minlora.get_lora_state_dict(raw_model)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint['best_val_loss'] = best_val_loss
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name))
        if eval_text and ppl < best_perplexity:
            best_perplexity = ppl
            checkpoint['best_perplexity'] = best_perplexity
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name.split('.pt')[0]+'_ppl.pt'))
        if eval_addition and test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint['best_accuracy'] = best_accuracy
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}/{ckpt_path_name}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name.split('.pt')[0]+'_acc.pt'))
        
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if save_final:
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name.split('.pt')[0]+'_final.pt'))

if ddp:
    destroy_process_group()
