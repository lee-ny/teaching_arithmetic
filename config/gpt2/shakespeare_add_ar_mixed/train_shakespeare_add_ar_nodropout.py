# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-gpt/shakespeare_add_ar_mixed'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

wandb_log = True # override via command line if you like
wandb_entity = 'ssdd'
wandb_project = 'gpt2-jt-shakespeare-add-ar-mixed'
wandb_run_name = 'gpt2-mixed'

dataset = 'shakespeare_add_ar_mixed'
train_data_path = 'train_all.bin' #'train_addition.bin'
val_data_path = 'val_all.bin' #'val_addition.bin'

eval_addition = True
start = 'FILE:data/addition_bal/prompt_addition_test_0.0001.txt'

eval_addition_ar = True
start_ar = "FILE:data/algo_reasoning/prompt_addition_test_0.0001.txt"

eval_text = True
eval_text_data_path = 'data/shakespeare_add_ar_mixed/val_text.bin' 


# these make the total batch size be ~0.5M 
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520 # TODO: Check this
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5

init_from = 'scratch'
ckpt_path_name = 'ckpt.pt'

max_iters = 50000
lr_decay_iters = 50000 # make equal to max_iters usually


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# python train.py config/train_shakespeare_addition_pad.py --wandb_log=False --ckpt_path_name="aaa.pt"