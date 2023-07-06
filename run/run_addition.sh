# training
### Task1
# python train.py config/train_shakespeare_char.py
# python train.py config/train_addition.py
# python train.py config/train_addition_pad.py
# python train.py config/finetune_shakespeare_addition_pad.py #--wandb_log=False
# python train.py config/train_shakespeare_addition_pad.py
# python train.py config/finetune_shakespeare.py

# Comparing with MC
# for num_train in {100,500,1000,1200,1500,1800,2000,5000}; do
for num_train in {1000,1200}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_mc.py \
    --wandb_project="addition" --wandb_run_name="mc_${num_train}" \
    --train_data_path="addition_train_100_${num_train}.txt" \
    --start="FILE:data/mc/addition_test_100_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
done


for num_train in {2000,5000}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_mc.py \
    --wandb_project="addition" --wandb_run_name="mc_${num_train}" \
    --train_data_path="addition_train_100_${num_train}.txt" \
    --start="FILE:data/mc/addition_test_100_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:1'
done

for num_train in {1000,1200}; do
    echo $num_train
    python config2/addition/reverse/train_addition_mc.py \
    --wandb_project="addition" --wandb_run_name="mc_reverse_${num_train}" \
    --train_data_path="addition_train_100_${num_train}.txt" \
    --start="FILE:data/mc/addition_test_100_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
done


for num_train in {2000,5000}; do
    echo $num_train
    python config2/addition/reverse/train_addition_mc.py \
    --wandb_project="addition" --wandb_run_name="mc_reverse_${num_train}" \
    --train_data_path="addition_train_100_${num_train}.txt" \
    --start="FILE:data/mc/addition_test_100_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:1'
done



# # comparing curriculum / balanced
# python evaluate_additions.py --wandb_project='addition' --wandb_run_name='eval_addition_bal' \
# --zero_pad=False --dataset='addition_bal' --out_dir='out/out-addition-bal' --ckpt_path_name='ckpt_10000_final.pt' \
# --prompt_overall="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:data/addition_bal/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/addition_bal/prompt_addition_test_2_0.1.txt"
# python train.py config/train_addition.py # no balancing
# python evaluate_additions.py --wandb_project='addition' --wandb_run_name='eval_addition_no_bal' \
# --zero_pad=False --dataset='addition' --out_dir='out/out-addition-no-bal' \
# --prompt_overall="FILE:data/addition/test_10000.txt" \
# --prompt1="FILE:prompt/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_test_2.txt"
# python train.py config/train_curriculum_no_balance.py
# python evaluate_additions.py --wandb_project='addition' --wandb_run_name='eval_addition_curriculum' \
# --zero_pad=False --dataset='addition' --out_dir='out/out-addition-curriculum' \
# --prompt_overall="FILE:data/addition/test_10000.txt" \
# --prompt1="FILE:prompt/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_test_2.txt"


# python train.py config/0222_train_addition_zero_padded.py
# python train.py config/0222_train_addition_curriculum.py --device='cuda:1'
# python train.py config/0222_train_addition_reverse.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='addition_char' \
# --reverse=False --zero_pad=False --dataset='addition_pad' --out_dir='out-addition-char-pad' \
# --prompt_overall="FILE:prompt/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_pad_test_2_0.1.txt"

# python evaluate_additions.py --wandb_run_name='addition_curriculum' \
# --reverse=False --zero_pad=False --dataset='addition_curriculum' --out_dir='0222-out-addition-curriculum' \
# --prompt_overall="FILE:prompt2/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt2/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt2/prompt_addition_pad_test_2_0.1.txt"

# python evaluate_additions.py --wandb_run_name='addition_zero_padded' \
# --reverse=False --zero_pad=True --dataset='addition_zero_pad' --out_dir='0222-out-addition-zero-padded' \
# --max_new_tokens=8 \
# --prompt_overall="FILE:prompt2/prompt_addition_pad_test_0.0001_zero_padded.txt" \
# --prompt1="FILE:prompt2/prompt_addition_pad_test_1_zero_padded.txt" \
# --prompt2="FILE:prompt2/prompt_addition_pad_test_2_0.1_zero_padded.txt" --wandb_log=False

# python evaluate_additions.py --wandb_run_name='addition_reverse' \
# --reverse=True --zero_pad=False --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --prompt_overall="FILE:prompt3/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt3/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt3/prompt_addition_pad_test_2_0.1.txt"


# python train.py config/0227_train_addition_pad_balanced.py

# python evaluate_additions.py --wandb_run_name='addition_balanced' \
# --reverse=False --zero_pad=False --dataset='addition_pad' --out_dir='0227-out-addition-balanced' \
# --prompt_overall="FILE:prompt/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_pad_test_2_0.1.txt"


# python train.py config/0227_train_addition_dollar.py

# python evaluate_additions.py --wandb_run_name='addition_$' --device='cuda:1' --wandb_log=False \
# --reverse=False --zero_pad=False --dataset='addition_dollar' --out_dir='0227-out-addition-dollar' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt_dollar/prompt_addition_pad_test_2_0.1.txt"


# python train.py config/0228_train_addition_dollar_curr_bal.py

# python evaluate_additions.py --wandb_run_name='addition_$_curr_bal' --device='cuda:1' \
# --reverse=False --zero_pad=False --dataset='addition_dollar_curr_bal' --out_dir='0228-out-addition-dollar-curr-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt_dollar/prompt_addition_pad_test_2_0.1.txt"

# python evaluate_additions.py --wandb_log=False \
# --zero_pad=False --dataset='addition_dollar_curr_bal' --out_dir='0228-out-addition-dollar-curr-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/train_prompt.txt"


# python evaluate_additions.py --wandb_run_name='addition_$' --device='cuda:1' --wandb_log=False \
# --reverse=False --zero_pad=False --dataset='addition_dollar' --out_dir='0227-out-addition-dollar' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/prompt_unit_test.txt" \
# --prompt1="FILE:prompt_dollar/prompt_unit_test.txt" \
# --prompt2="FILE:prompt_dollar/prompt_unit_test.txt"


# python train.py config/train_addition.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='addition_no_pad' --device='cuda:0' \
# --reverse=False --zero_pad=False --dataset='addition' --out_dir='out-addition-char' \
# --prompt_overall="FILE:prompt/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:prompt/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_test_2_0.1.txt"


# python train.py config/0301_train_addition_dollar_reverse_curr_bal.py --device='cuda:1'

# python evaluate_additions.py --wandb_run_name='addition_dollar_reverse_curr_bal' --device='cuda:1' --wandb_log=False \
# --reverse_ab=True --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal' --out_dir='0301-out-addition-dollar-reverse-curr-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar_reverse/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt_dollar_reverse/prompt_addition_pad_test_2_0.1.txt"

# python train.py config/0301_train_addition_dollar_reverse_curr_bal2.py --device='cuda:0'

# python evaluate_additions.py --wandb_run_name='addition_dollar_reverse_curr_bal2' --device='cuda:1' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.1.txt" \
# --prompt1="FILE:prompt_dollar_reverse2/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt_dollar_reverse2/prompt_addition_test_2_0.1.txt"

# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_nonoverlap.txt" \

# python evaluate_additions.py --wandb_log=False \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='0301-out-addition-dollar-reverse-curr-bal2' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar_reverse2/train_prompt.txt"



# for num_train in {100,500,1000,2000,5000}; do
#     echo $num_train
#     python train.py config/0308_train_addition_dollar_reverse_curr_bal2_4digit.py --device='cuda:1' \
#     --wandb_run_name="$-reverse-curr-bal2-4digit_${num_train}" \
#     --train_data_path="train_4digit_${num_train}.bin" --val_data_path="val_4digit_${num_train}.bin" \
#     --ckpt_path_name="ckpt_4digit_${num_train}.pt"
# done

### python train.py config/0308_train_addition_dollar_reverse_curr_bal2_4digit.py --device='cuda:1'


# python train.py config/0321_train_addition_dollar_curr_bal_larger.py

# for lr in {1e-2,1e-3,1e-4,1e-5}; do # Best: 1e-3
# for lr in {1e-3,}; do # Best: 1e-3
#     echo $lr
#     python train.py config/0321_train_addition_dollar_curr_bal_larger.py \
#     --wandb_run_name="addition-$-curr-bal_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done


# python evaluate_additions.py --wandb_run_name='eval_addition_$_curr_bal' --device='cuda:1' \
# --dataset='addition_dollar_curr_bal' \
# --out_dir='out/addition-dollar_bal_large' --ckpt_path_name='ckpt_1e-4_final.pt' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt_dollar/prompt_addition_pad_test_2_0.1.txt"

# python evaluate_additions.py --wandb_log=False \
# \--dataset='addition_dollar_curr_bal' \
# --out_dir='out/addition-dollar_bal_large' --ckpt_path_name='ckpt_1e-3_final.pt' \
# --max_new_tokens=5 \
# --prompt_overall="FILE:prompt_dollar/train_prompt.txt"
#### accuracy of 9064 examples: 8983/9064 (99.1063548102383%)
#### {'carry0': 98.56, 'carry1': 98.8, 'carry2': 99.44, 'carry3': 99.93606138107417}




# python sample_addition.py --wandb_run_name='addition_char' --ckpt_path_name='ckpt_final.pt' \
# --reverse=False --zero_pad=False --dataset='addition_pad' --out_dir='out-addition-char-pad' \
# --start="FILE:prompt/prompt_addition_pad_test_0.01.txt" > char.log

# python sample_addition.py --wandb_run_name='addition_curriculum' --ckpt_path_name='ckpt_final.pt' \
# --reverse=False --zero_pad=False --dataset='addition_curriculum' --out_dir='0222-out-addition-curriculum' \
# --start="FILE:prompt2/prompt_addition_pad_test_0.01.txt" > curriculum.log

# python sample_addition.py --wandb_run_name='addition_zero_padded' --ckpt_path_name='ckpt_final.pt' \
# --reverse=False --zero_pad=True --dataset='addition_zero_pad' --out_dir='0222-out-addition-zero-padded' \
# --max_new_tokens=8 \
# --start="FILE:prompt2/prompt_addition_pad_test_0.01_zero_padded.txt" > zero_pad.log

# python sample_addition.py --wandb_run_name='addition_reverse' --ckpt_path_name='ckpt_final.pt' \
# --reverse=True --zero_pad=False --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start="FILE:prompt3/prompt_addition_pad_test_0.01.txt" > reverese.log


### evaluating reverse1 (a,b,c reversed)
# 3160+1737=4897
# -> 0613+7371=7984
# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='0613+7371=' 
# 0613+7371=7980

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='1642+1684=2237
# 0613+7371=' 
# 0613+7371=7978

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='1642+1684=2237
# 3490+1480=4871
# 0613+7371=' 
# 0613+7371=7631

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='1642+1684=2237
# 3490+1480=4871
# 9242+3481=2724
# 0613+7371=' 
# 0613+7371=0059

# 2461+4861=7322
# -> 1642+1684=2237
# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='1642+1684=' 
# 1642+1684=2221

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='6213+8720=4043
# 360+7561=0271
# 1642+1684=' 
# 1642+1684=3229

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='6213+8720=4043
# 3600+7561=0271
# 1642+1684=' 
# 1642+1684=3010

# python sample.py  --ckpt_path_name='ckpt_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --dataset='addition_pad_reverse' --out_dir='0222-out-addition-reverse' \
# --start='1642+1684=2237
# 1642+1684=' 
# 1642+1684=3212



# for num_train in {13000,26000,39000,52000,65000,78000,91000,104000,117000}; do
#     echo $num_train
#     python train.py config/train_addition_pad.py \
#     --wandb_run_name="mini-gpt-padded_${num_train}" \
#     --train_data_path="train_${num_train}.bin" --val_data_path="val_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


##############################################
##############################################

### testing Addition
# python sample_addition.py --start="FILE:prompt/prompt_addition_test_0.001.txt" --out_dir=out-addition-char
# python sample_addition.py --start="FILE:prompt/prompt_addition_pad_test_0.01.txt" --out_dir=out-addition-char-pad --plot_sample_acc=True --wandb_run_name='num_train-mini-gpt-padded_0.01'

# python sample_addition.py --device='cuda:1' --wandb_log=False --start="FILE:prompt/prompt_addition_pad_test_0.0001.txt" --out_dir=out-ft-shakespeare-addition-char-pad --plot_sample_acc=True --wandb_run_name='num_train-mini-gpt-padded_0.0001' --ckpt_path_name='ckpt_1e-3.pt'

# python sample_addition.py --start="FILE:prompt/prompt_addition_pad_test_0.01.txt" \
# --out_dir=out-ft-shakespeare-addition-char-pad --plot_sample_acc=True --dataset='shakespeare_addition_char' \
# --wandb_project='ft-shakespeare-addition-char' --wandb_run_name='num_train-mini-gpt-padded_0.01'




##### In-context
# python sample.py  --out_dir=out-addition-char-pad --dataset=addition_pad --evaluate=False --num_samples=1 --max_new_tokens=4 --start='121+1555=1676                                               
# 3152+1431=4583
# 4843+249=5092
# 321+1342='

# python sample.py  --out_dir=out-ft-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-4.pt --dataset=shakespeare_addition_char --evaluate=False --num_samples=1 --max_new_tokens=4 --start='121+1555=1676
# 3152+1431=4583
# 4843+249=5092
# 321+1342='

# python sample.py  --out_dir=out-jt-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-3.pt --dataset=shakespeare_addition_char --evaluate=False --num_samples=1 --max_new_tokens=4 --start='121+1555=1676
# 3152+1431=4583
# 4843+249=5092
# 321+1342='

