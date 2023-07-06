import time

out_dir = 'out-addition'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'addition'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare_addition'
train_data_path='train_addition.bin' #'train_addition.bin'
val_data_path='val_addition.bin' #'val_addition.bin'
init_from = 'gpt2'
eval_text = True
eval_text_data_path = 'data/shakespeare_addition/test_addition.bin'
eval_addition = True
start = "FILE:prompt/prompt_addition_pad_test_0.0001.txt"

# only save checkpoints if the validation loss improves
always_save_checkpoint = False
save_final = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 500

# finetune at constant LR
learning_rate = 1e-4 # 3e-5
decay_lr = False
