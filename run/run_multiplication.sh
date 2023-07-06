# training
### Task1

# python train.py config/multiplication/plain/train_addition_bal.py --device='cuda:0' 

# python evaluate_additions.py --wandb_project='multiplication' --wandb_run_name='eval_plain' --device='cuda:1' \
# --dataset='multiplication/plain' --out_dir='out-multiplication/plain' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/multiplication/plain/train_examples_3000_test.txt" > out-multiplication/plain/eval_plain.txt

# python train.py config/multiplication/dollar_reverse/train_addition_dollar_reverse.py --device='cuda:1'

# python evaluate_additions.py --wandb_project='multiplication' --wandb_run_name='eval_reverse' --device='cuda:1' \
# --dataset='multiplication/dollar_reverse' --out_dir='out-multiplication/dollar_reverse' \
# --reverse_c=True --zero_pad=False --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/multiplication/dollar_reverse/train_examples_3000_test.txt" > out-multiplication/dollar_reverse/eval_reverse.txt


python train.py config2/multiplication/plain/train_addition_bal.py --device='cuda:0'

python train.py config2/multiplication/dollar_reverse/train_addition_dollar_reverse.py

python train.py config2/multiplication/algorithmic_reasoning/train_addition_ar.py

python evaluate_additions.py --wandb_project='multiplication' --wandb_run_name='eval_ar' --device='cuda:1' \
--dataset='bal' --out_dir='out2/multiplication_ar' \
--algo_reason=True --verbose=True \
--data_type='text' --data_format='algo_reasoning' --operator='*' \
--prompt_overall="FILE:data/bal/test_multiplication_7000.txt" \
--prompt1="FILE:data/bal/train_multiplication_3000.txt" > out2/multiplication_ar/eval_ar.txt
