# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/shakespeare_add_ar_mixed'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_entity = 'ssdd'
wandb_project = 'jt-shakespeare-add-ar-mixed'
wandb_run_name = 'mini-gpt-padded-mixed'

dataset = 'shakespeare_add_ar_mixed'
train_data_path = 'train_all.bin' #'train_addition.bin'
val_data_path = 'val_all.bin' #'val_addition.bin'

# train_both = True
# data_ratio = 0.2 # ratio of data_path2 compared with data_path1
# train_data_path2 = 'train_addition.bin' # only used when train_both = True
# val_data_path2 = 'val_addition.bin'

eval_addition = True
start = 'FILE:data/addition_bal/prompt_addition_test_0.0001.txt'

eval_addition_ar = True
start_ar = "FILE:data/algo_reasoning/prompt_addition_test_0.0001.txt"

eval_text = True
eval_text_data_path = 'data/shakespeare_add_ar_mixed/val_text.bin' 

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

batch_size = 16
block_size = 1024 # context of up to 256 previous characters

init_from = 'scratch'
ckpt_path_name = 'ckpt.pt'

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 50000
lr_decay_iters = 50000 # make equal to max_iters usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

device='cuda:0'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# python train.py config/train_shakespeare_addition_pad.py --wandb_log=False --ckpt_path_name="aaa.pt"