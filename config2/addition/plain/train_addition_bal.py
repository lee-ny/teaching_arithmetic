# train plain addition on NanoGPT

# ===== Evaluation and Checkpointing ===== #
out_dir = 'out2/addition_plain'
eval_interval = 250 
eval_iters = 200
log_interval = 10 
always_save_checkpoint = False

# ===== Wandb logging ===== #
wandb_log = True # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'addition_plain'

# ===== Dataset ===== #
data_type='text'
data_format='plain'
operator='+'
dataset = 'bal'
batch_size = 256
block_size = 256 # context of up to 256 previous characters
train_data_path = 'train_3digit_10000.txt'
ckpt_path_name = 'ckpt_10000.pt'
eval_addition = True
start = "FILE:data/bal/test_10000.txt"
eval_addition_train = True

# ===== NanoGPT model configuration ===== #
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# ===== Learning Rate Policy ===== #
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
beta2 = 0.99
warmup_iters = 100

# ===== Device ===== #
device='cuda:0'