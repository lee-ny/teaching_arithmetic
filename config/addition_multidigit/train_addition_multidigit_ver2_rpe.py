# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-rpe/addition_multidigit_ver2'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

multi_digit = True
num_digit = 7
dataset = 'addition_multidigit_bal_ver2'
train_data_path = f'train_{num_digit}.bin' #'train_addition.bin'
val_data_path = f'val_{num_digit}.bin' #'val_addition.bin'

# train_both = True
# data_ratio = 0.2 # ratio of data_path2 compared with data_path1
# train_data_path2 = 'train_addition.bin' # only used when train_both = True
# val_data_path2 = 'val_addition.bin'

eval_addition = True
start = f"FILE:data/{dataset}/prompt_test_{num_digit}digit_100.txt"
reverse_c = True

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

batch_size = 128
block_size = 256 # context of up to 256 previous characters


learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

device='cuda:0'

init_from = 'scratch'
ckpt_path_name = f'ckpt_{num_digit}digit_lr{learning_rate}.pt'

wandb_log = True # override via command line if you like
wandb_entity = 'ssdd'
wandb_project = 'addition_multidigit'
wandb_run_name = f'rpe-ver2_{num_digit}digit_lr{learning_rate}'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# python train.py config/train_shakespeare_addition_pad.py --wandb_log=False --ckpt_path_name="aaa.pt"