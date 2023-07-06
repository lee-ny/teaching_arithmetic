import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
# train_data_path='train_shakespeare.bin' #'train_addition.bin'
# val_data_path='val_shakespeare.bin' #'val_addition.bin'
# init_from = 'gpt2'
init_from = 'gpt2-large' # use a smaller for faster training

# eval_text = True
# eval_text_data_path = 'data/shakespeare_addition/test_shakespeare.bin'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False
save_final = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
# max_iters = 500
max_iters=20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
