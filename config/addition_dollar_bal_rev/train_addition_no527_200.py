# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/out-addition-dollar-reverse-bal-no527_200'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'reverse, no 200 num'

dataset = 'addition_dollar_reverse_curr_bal2'
batch_size = 256
block_size = 256 # context of up to 256 previous characters
train_data_path = 'train_no527_200.bin' # NOTE: this contains the val.bin data since we did not split
val_data_path = 'val.bin'
ckpt_path_name = 'ckpt.pt'
reverse_c = True
eval_addition = True
start = "FILE:prompt_dollar_reverse2/prompt_addition_test_0.0001.txt"
eval_addition_527 = True
start_527 = "FILE:data/addition_dollar_reverse_curr_bal2/527_200_test.txt"

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
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
