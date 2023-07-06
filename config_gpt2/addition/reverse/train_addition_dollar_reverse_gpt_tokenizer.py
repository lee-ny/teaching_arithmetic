# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out2-gpt/addition_reverse_gpt_tokenizer'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'addition-gpt'
wandb_run_name = 'reverse_gpt_tokenizer'

data_type='text'
data_format='reverse'
operator='+'
dataset = 'bal'
batch_size = 8
block_size = 1024 # context of up to 256 previous characters
train_data_path = 'train_3digit_10000.txt'
# val_data_path = 'val.bin'
ckpt_path_name = 'ckpt.pt'
reverse_c = True
eval_addition = True
start = "FILE:data/bal/test_10000.txt"
eval_addition_train = True
# start_train = "FILE:data/one-sided-subtraction/dollar_reverse/add_examples_trainprompt.txt"
tokenizer = 'gpt2'

gradient_accumulation_steps = 5
dropout = 0.2


learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device='cuda:0'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
