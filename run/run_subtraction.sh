# training
### Task1

# python train.py config/one-sided-subtraction/plain/train_addition_bal.py --device='cuda:0' 

# python evaluate_additions.py --wandb_project='subtraction' --wandb_run_name='eval_plain' --device='cuda:1' \
# --dataset='one-sided-subtraction/plain' --out_dir='out-one-sided-subtraction/plain' \
# --max_new_tokens=4 \
# --prompt_overall="FILE:data/one-sided-subtraction/plain/prompt_addition_test_0.01.txt"

# python train.py config/one-sided-subtraction/dollar_reverse/train_addition_dollar_reverse.py --device='cuda:1'

# python evaluate_additions.py --wandb_project='subtraction' --wandb_run_name='eval_reverse' --device='cuda:1' \
# --dataset='one-sided-subtraction/dollar_reverse' --out_dir='out-one-sided-subtraction/dollar_reverse' \
# --reverse_c=True --zero_pad=False --max_new_tokens=4 \
# --prompt_overall="FILE:data/one-sided-subtraction/dollar_reverse/prompt_addition_test_0.01.txt"


python train.py config2/subtraction/plain/train_addition_bal.py --device='cuda:1' 

python train.py config2/subtraction/dollar_reverse/train_addition_dollar_reverse.py --device='cuda:0'

python train.py config2/subtraction/algorithmic_reasoning/train_addition_ar.py --device='cuda:0'

python train.py config2/subtraction/algorithmic_reasoning/train_addition_ar.py --device='cuda:1' --wandb_run_name='algo_reasoning_v2'
