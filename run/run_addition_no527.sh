

####################################################################
#####################        no 527          #######################
###################################################################@

# python train_no527.py config/addition_bal/train_addition_no527.py --device='cuda:0'
# python train.py config/addition_bal/train_addition_bal.py --train_data_path='train_10000.bin' --wandb_run_name='add-bal_10000'

# python evaluate_additions.py --wandb_run_name='no527-add-bal_10000_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no527' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_test.txt"

# python evaluate_additions.py --wandb_run_name='add-bal_10000_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_test.txt"


# python train_no527.py config/addition_dollar_bal_rev/train_addition_no527.py --device='cuda:0'

# python evaluate_additions.py --wandb_run_name='addition-$-reverse-curr-bal2_eval' --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_test.txt"

# python evaluate_additions.py --wandb_run_name='no527-reverse-curr-bal2_eval' --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no527' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_test.txt" \


#########
#### eval on vanilla addition / reverse

# python evaluate_additions.py --wandb_run_name='add-bal_10000_eval_no527' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/527_test.txt" \
# --prompt1="FILE:data/addition_bal/527_10_test.txt" \
# --prompt2="FILE:data/addition_bal/527_100_test.txt"

# --prompt_overall="FILE:data/addition_bal/527_200_test.txt" \
# --prompt1="FILE:data/addition_bal/527_500_test.txt"

# python evaluate_additions.py --wandb_run_name='addition-$-reverse-curr-bal2_eval_no527' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/527_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_10_test.txt" \
# --prompt2="FILE:data/addition_dollar_reverse_curr_bal2/527_100_test.txt"

# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/527_200_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_500_test.txt"


#########

# python train_no527.py config/addition_bal/train_addition_no527_10.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='no527_10-add-bal_10000_eval' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no527_10' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_10_test.txt"


# python train_no527.py config/addition_dollar_bal_rev/train_addition_no527_10.py --device='cuda:1'


# python evaluate_additions.py --wandb_run_name='no527_10-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no527_10' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_10_test.txt" \


#########

# python train_no527.py config/addition_bal/train_addition_no527_100.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='no527_100-add-bal_10000_eval' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no527_100' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_100_test.txt"


# python train_no527.py config/addition_dollar_bal_rev/train_addition_no527_100.py --device='cuda:1'


# python evaluate_additions.py --wandb_run_name='no527_100-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no527_100' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/527_100_test.txt" \


# testing on 4 digit prompts
# python evaluate_additions.py --wandb_log=False --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no527_100' \
# --max_new_tokens=6 --num_digit=4 \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/prompt_4digit_100.txt" --verbose=True > out/out-addition-dollar-reverse-bal-no527_100/prompt_4digit_100.txt

# python evaluate_additions.py --wandb_log=False --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --max_new_tokens=6 --num_digit=4 \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/prompt_4digit_100.txt" --verbose=True > out/out-addition-dollar-reverse-bal/prompt_4digit_100.txt






#########
### having number 5 not appear in some digit places

######
## no 5 in the 2nd digit
######
### evaluating addition, reverse model

# python evaluate_additions.py --wandb_run_name='add-bal_10000_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_bal/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_bal/no5_3rddigit_test.txt" --verbose=False

# python evaluate_additions.py --wandb_run_name='addition-$-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_dollar_reverse_curr_bal2/no5_3rddigit_test.txt" --verbose=False

###############

# python train_no527.py config/addition_bal/train_addition_no5_2nddigit.py --device='cuda:0'

# python evaluate_additions.py --wandb_run_name='no5-2nddigit-add-bal_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-2nddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/no5_2nddigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-2nddigit-add-bal_eval' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-2nddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_bal/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_bal/no5_3rddigit_test.txt" --verbose=False

# python train_no527.py config/addition_dollar_bal_rev/train_addition_no5_2nddigit.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='no5-2nddigit-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-2nddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_2nddigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-2nddigit-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-2nddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_dollar_reverse_curr_bal2/no5_3rddigit_test.txt" --verbose=False

#################

# python train_no527.py config/addition_bal/train_addition_no5_1stdigit.py --device='cuda:0'

# python evaluate_additions.py --wandb_run_name='no5-1stdigit-add-bal_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-1stdigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/no5_1stdigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-1stdigit-add-bal_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-1stdigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_bal/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_bal/no5_3rddigit_test.txt" --verbose=False


# python train_no527.py config/addition_dollar_bal_rev/train_addition_no5_1stdigit.py --device='cuda:0'

# python evaluate_additions.py --wandb_run_name='no5-1stdigit-reverse-curr-bal2_eval' --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-1stdigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_1stdigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-1stdigit-reverse-curr-bal2_eval' --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-1stdigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_dollar_reverse_curr_bal2/no5_3rddigit_test.txt" --verbose=False


## evaluating on train set  - does it overfit? YES!
# python evaluate_additions.py --wandb_run_name='no5-1stdigit-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-1stdigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/prompt_no5_1stdigit_train.txt" --wandb_log=False


####################

# python train_no527.py config/addition_bal/train_addition_no5_3rddigit.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='no5-3rddigit-add-bal_eval' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-3rddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/no5_3rddigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-3rddigit-add-bal_eval' --device='cuda:1' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no5-3rddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_bal/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_bal/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_bal/no5_3rddigit_test.txt" --verbose=False


# python train_no527.py config/addition_dollar_bal_rev/train_addition_no5_3rddigit.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='no5-3rddigit-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-3rddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_3rddigit_test.txt"

# python evaluate_additions.py --wandb_run_name='no5-3rddigit-reverse-curr-bal2_eval' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal-no5-3rddigit' \
# --max_new_tokens=5 --verbose=True \
# --prompt_overall="FILE:data/addition_dollar_reverse_curr_bal2/no5_1stdigit_test.txt" \
# --prompt1="FILE:data/addition_dollar_reverse_curr_bal2/no5_2nddigit_test.txt" \
# --prompt2="FILE:data/addition_dollar_reverse_curr_bal2/no5_3rddigit_test.txt" --verbose=False


####################################################################
##################  small sample/ small pool  ######################
####################################################################

python train_no527.py config/addition_bal/train_addition_smallsample.py --device='cuda:0'
python train_no527.py config/addition_bal/train_addition_smallpool.py --device='cuda:0'

python train_no527.py config/addition_dollar_bal_rev/train_addition_smallpool.py --device='cuda:1'
python train_no527.py config/addition_dollar_bal_rev/train_addition_smallsample.py --device='cuda:1'


# python evaluate_additions.py --wandb_run_name='no527-add-bal_10000_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal-no527' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_test.txt"

# python evaluate_additions.py --wandb_run_name='add-bal_10000_eval' --device='cuda:0' \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/527_test.txt"