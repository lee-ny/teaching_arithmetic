###############
####  sin #####
###############

# python train.py config2/sin/train_addition_bal.py --device='cuda:0'

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_eps5e-3' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin/eval_test_sin.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_eps5e-3' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin/eval_train_sin.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_eps5e-4' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin/eval_test_sin.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_eps5e-4' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin/eval_train_sin.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin(exact)' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin/eval_test_sin_exact.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin(exact)' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin' \
# --data_type='text' --operator='sin' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin/eval_train_sin_exact.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_overall' --device='cuda:1' \
--dataset='sin' --out_dir='out2/sin' \
--data_type='text' --operator='sin' \
--num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
--prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin/eval_train_sin_overall.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_overall' --device='cuda:1' \
--dataset='sin' --out_dir='out2/sin' \
--data_type='text' --operator='sin' \
--num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
--prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin/eval_test_sin_overall.txt

## AR ##

# python train.py config2/sin/train_ar.py --device='cuda:0'

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_ar' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin_ar/eval_test_sin_ar.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_ar' --device='cuda:1' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin_ar/eval_train_sin_ar.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_ar_eps5e-3' --device='cuda:0' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin_ar/eval_test_sin_ar_5e-3.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_ar_eps5e-3' --device='cuda:0' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin_ar/eval_train_sin_ar_5e-3.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_ar_eps5e-4' --device='cuda:0' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin_ar/eval_test_sin_ar_5e-4.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_ar_eps5e-4' --device='cuda:0' \
# --dataset='sin' --out_dir='out2/sin_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sin' --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin_ar/eval_train_sin_ar_5e-4.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sin_ar_overall' --device='cuda:1' \
--dataset='sin' --out_dir='out2/sin_ar' \
--algo_reason=True --verbose=True \
--data_type='text' --data_format='algo_reasoning' --operator='sin' --analyze=True \
--prompt_overall="FILE:data/sin/test_sin_10000.txt" > out2/sin_ar/eval_test_sin_ar_overall.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sin_ar_overall' --device='cuda:1' \
--dataset='sin' --out_dir='out2/sin_ar' \
--algo_reason=True --verbose=True \
--data_type='text' --data_format='algo_reasoning' --operator='sin' --analyze=True \
--prompt_overall="FILE:data/sin/train_sin_10000.txt" > out2/sin_ar/eval_train_sin_ar_overall.txt



###############
#### sqrt #####
###############

# python train.py config2/sqrt/train_addition_bal.py --device='cuda:1'

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_eps5e-3' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt/eval_test_sqrt_5e-3.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_eps5e-3' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt/eval_train_sqrt_5e-3.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_eps5e-4' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt/eval_test_sqrt_eps5e-4.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_eps5e-4' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt/eval_train_sqrt_eps5e-4.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt(exact)' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt/eval_test_sqrt_exact.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt(exact)' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt' \
# --data_type='text' --operator='sqrt' \
# --num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt/eval_train_sqrt_exact.txt


python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_overall' --device='cuda:0' \
--dataset='sqrt' --out_dir='out2/sqrt' \
--data_type='text' --operator='sqrt' \
--num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
--prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt/eval_test_sqrt_overall.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_overall' --device='cuda:0' \
--dataset='sqrt' --out_dir='out2/sqrt' \
--data_type='text' --operator='sqrt' \
--num_digit=5 --max_new_tokens=8 --verbose=True --eps=0.0 --analyze=True \
--prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt/eval_train_sqrt_overall.txt

## AR ##

# python train.py config2/sqrt/train_ar.py --device='cuda:1'

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_ar' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt_ar/eval_test_sqrt_ar.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_ar' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt_ar/eval_train_sqrt_ar.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_ar_eps5e-3' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt_ar/eval_test_sqrt_ar_5e-3.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_ar_eps5e-3' --device='cuda:1' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --eps=5e-3 --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt_ar/eval_train_sqrt_ar_5e-3.txt


# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_ar_eps5e-4' --device='cuda:0' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt_ar/eval_test_sqrt_ar_5e-4.txt

# python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_ar_eps5e-4' --device='cuda:0' \
# --dataset='sqrt' --out_dir='out2/sqrt_ar' \
# --algo_reason=True --verbose=True \
# --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --eps=5e-4 --analyze=True \
# --prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt_ar/eval_train_sqrt_ar_5e-4.txt


python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_test_sqrt_ar_overall' --device='cuda:0' \
--dataset='sqrt' --out_dir='out2/sqrt_ar' \
--algo_reason=True --verbose=True \
--data_type='text' --data_format='algo_reasoning' --operator='sqrt' --analyze=True \
--prompt_overall="FILE:data/sqrt/test_sqrt_10000.txt" > out2/sqrt_ar/eval_test_sqrt_ar_overall.txt

python evaluate_additions.py --wandb_project='sin_sqrt' --wandb_run_name='eval_train_sqrt_ar_overall' --device='cuda:0' \
--dataset='sqrt' --out_dir='out2/sqrt_ar' \
--algo_reason=True --verbose=True \
--data_type='text' --data_format='algo_reasoning' --operator='sqrt' --analyze=True \
--prompt_overall="FILE:data/sqrt/train_sqrt_10000.txt" > out2/sqrt_ar/eval_train_sqrt_ar_overall.txt

