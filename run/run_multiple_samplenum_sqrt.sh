# python train.py config/algo_reasoning/train_addition.py --wandb_run_name='lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'


##################################
############ add Bal ###########
##################################


for num_train in {1000,3000,5000,20000,40000}; do
    echo $num_train
    python train.py config2/sqrt/train_addition_bal.py \
    --wandb_run_name="sqrt_plain_${num_train}" \
    --train_data_path="train_sqrt_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt"
done

python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_plain_Zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='plain' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt' \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --device='cuda:0'


python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_plain_Zeroshot_eps_5e_4' \
--plot_sample_acc=True --select='samplenum' --eps=5e-4 \
--data_type='text' --data_format='plain' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt' \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --device='cuda:0'

python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_plain_Zeroshot_eps_5e_3' \
--plot_sample_acc=True --select='samplenum' --eps=5e-3 \
--data_type='text' --data_format='plain' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt' \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --device='cuda:1'



for num_train in {1000,3000,5000,20000,40000}; do
    echo $num_train
    python train.py config2/sqrt/train_ar.py \
    --wandb_run_name="sqrt_ar_${num_train}" \
    --train_data_path="train_sqrt_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:1'
done

python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_ar_Zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--algo_reason=True --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt_ar' \
--start="FILE:data/sqrt/test_sqrt_1000.txt" --device='cuda:1'


python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_ar_Zeroshot_eps_5e_4' \
--plot_sample_acc=True --select='samplenum' --eps=5e-4 \
--algo_reason=True --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt_ar' \
--start="FILE:data/sqrt/test_sqrt_1000.txt" --device='cuda:0'


python sample_addition.py \
--wandb_log=True --wandb_project='sin_sqrt' --wandb_run_name='eval_sqrt_ar_Zeroshot_eps_5e_3' \
--plot_sample_acc=True --select='samplenum' --eps=5e-3 \
--algo_reason=True --data_type='text' --data_format='algo_reasoning' --operator='sqrt' --num_digit=5 --max_new_tokens=8 \
--dataset='sqrt' --out_dir='out2/sqrt_ar' \
--start="FILE:data/sqrt/test_sqrt_1000.txt" --device='cuda:1'

# for num_train in {1000,2000,5000,7000}; do
#     echo $num_train
#     python train.py config2/multiplication/dollar_reverse/train_addition_dollar_reverse.py \
#     --wandb_run_name="reverse_${num_train}" \
#     --train_data_path="train_multiplication_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='multiplication' --wandb_run_name='eval_reverse_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='reverse' --operator='*' --reverse_c=True \
# --dataset='bal' --out_dir='out2/multiplication_reverse' \
# --start="FILE:data/bal/test_multiplication2_3000.txt" --device='cuda:0'

# for num_train in {1000,2000,5000,7000}; do
#     echo $num_train
#     python train.py config2/multiplication/algorithmic_reasoning/train_addition_ar.py \
#     --wandb_run_name="ar_${num_train}" \
#     --train_data_path="train_multiplication_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='multiplication' --wandb_run_name='eval_ar_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='*' --algo_reason=True \
# --dataset='bal' --out_dir='out2/multiplication_ar' \
# --start="FILE:data/bal/test_multiplication2_1000.txt" --device='cuda:0'


# # # few-shot! evaluate different numbers of addition examples
# for shots in {1,2,3}; do
#     python sample_addition_fewshot.py \
#     --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_add_${shots}shot" \
#     --dataset='addition_bal' --out_dir='out/out-addition-bal' \
#     --plot_sample_acc=True --select='samplenum' \
#     --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_${shots}shot_test_0.01_1.txt" \
#     --fewshot=True --device='cuda:0'
# done

## running evaluation for just 1 ckpt (num_sample=40000)
# for shots in {1,2,3}; do
#     python sample_addition_fewshot.py \
#     --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_add_${shots}shot_40000" \
#     --dataset='addition_bal' --out_dir='out/out-addition-bal' --ckpt_path_name='ckpt_40000_final.pt' \
#     --select='' \
#     --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_${shots}shot_test_0.01_1.txt" \
#     --fewshot=True --device='cuda:1'
# done


# evaluation on noisy prompt
# python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_add_noisyprompt" \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --plot_sample_acc=True --select='samplenum' \
# --start="FILE:data/addition_bal/few_shot_noisy_prompts/prompt_addition_3shot_test_0.01_1.txt" \
# --fewshot=True --device='cuda:1'



# ## evaluation on 4-digit addition
# python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_add_4digit_prompt" \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --plot_sample_acc=True --select='samplenum' --max_new_tokens=5 --num_digit=4 \
# --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_4digit_3shot_test_1.txt" \
# --fewshot=True --device='cuda:0'

# python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_add_4digit_prompt(20-shot)" \
# --dataset='addition_bal' --out_dir='out/out-addition-bal' \
# --plot_sample_acc=True --select='samplenum' --max_new_tokens=5 --num_digit=4 \
# --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_4digit_20shot_test_1.txt" \
# --fewshot=True --device='cuda:0'



##################################
############ $ Rev Bal ###########
##################################

### python train.py config/0301_train_addition_dollar_reverse_curr_bal2.py --device='cuda:0'
## python train.py config/0301_train_addition_dollar_reverse_curr_bal2.py --wandb_run_name="$-rev-bal_10000" --train_data_path="train.bin" --ckpt_path_name="ckpt_10000.pt" --device='cuda:1'

# for num_train in {2000,5000,20000}; do
# for num_train in {1000,3000,4000}; do
#     echo $num_train
#     python train.py config/0301_train_addition_dollar_reverse_curr_bal2.py \
#     --wandb_run_name="$-rev-bal_${num_train}" \
#     --train_data_path="train_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


# for num_train in {1000,3000,4000}; do
#     echo $num_train
#     python train.py config/0301_train_addition_dollar_reverse_curr_bal2.py \
#     --wandb_run_name="$-rev-bal_${num_train}" \
#     --train_data_path="train_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


# python sample_addition.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name='eval_$-rev-bal_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --start="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt"


# evaluation on 4-digit addition
# python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name="eval_reverse_4digit_prompt" \
# --reverse_c=True --zero_pad=False --dataset='addition_dollar_reverse_curr_bal2' --out_dir='out/out-addition-dollar-reverse-bal' \
# --plot_sample_acc=True --select='samplenum' --max_new_tokens=6 --num_digit=4 \
# --start="FILE:data/addition_dollar_reverse_curr_bal2/few_shot_prompts/prompt_addition_4digit_3shot_test_1.txt" \
# --fewshot=True --device='cuda:1'




####################################
############ Algo Reason ###########
####################################

### python train.py config/algo_reasoning/train_addition.py --device='cuda:1'
## python train.py config/algo_reasoning/train_addition.py --wandb_project="addition" --wandb_run_name="AR_10000" --train_data_path="train_balanced.bin" --ckpt_path_name="ckpt_10000.pt" --device='cuda:0'

# for num_train in {2000,5000,20000}; do
# for num_train in {1000,3000,4000}; do
#     echo $num_train
#     python train.py config/algo_reasoning/train_addition.py \
#     --wandb_project="addition" --wandb_run_name="AR_${num_train}" \
#     --train_data_path="train_balanced_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


# python sample_addition.py \
# --wandb_log=True --wandb_project='addition' --wandb_run_name='eval_AR_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --algo_reason=True --dataset='algo_reasoning' --out_dir='out/algo_reasoning-addition' \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt"


####################################
############# Simple AR ############
####################################

# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config/ar_simple/train_addition.py \
#     --wandb_run_name="simple_${num_train}" \
#     --train_data_path="train_balanced_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project="algo_reasoning" --wandb_run_name='eval_simple_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --algo_reason=True --dataset='ar_simple' --out_dir='out/ar_simple' \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt"


####################################
############ Simple AR-A ###########
####################################

# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config/ar_simple/train_addition_randomA.py \
#     --wandb_run_name="simple_randomA_${num_train}" \
#     --train_data_path="train_balanced_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project="algo_reasoning" --wandb_run_name='eval_simple_randomA_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --algo_reason=True --dataset='ar_simple_randomA' --out_dir='out/ar_simple_randomA' \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt"

####################################
############ Simple AR-C ###########
####################################

# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config/ar_simple/train_addition_randomC.py \
#     --wandb_run_name="simple_randomC_${num_train}" \
#     --train_data_path="train_balanced_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project="algo_reasoning" --wandb_run_name='eval_simple_randomC_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --algo_reason=True --dataset='ar_simple_randomC' --out_dir='out/ar_simple_randomC' \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt"


# ####################################
# ########### Simple AR-AC ###########
# ####################################

# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config/ar_simple/train_addition_randomboth.py \
#     --wandb_run_name="simple_randomBoth_${num_train}" \
#     --train_data_path="train_balanced_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project="algo_reasoning" --wandb_run_name='eval_simple_randomBoth_samplenum' \
# --plot_sample_acc=True --select='samplenum' \
# --algo_reason=True --dataset='ar_simple_randomboth' --out_dir='out/ar_simple_randomBoth' \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt"
