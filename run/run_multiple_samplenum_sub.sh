# python train.py config/algo_reasoning/train_addition.py --wandb_run_name='lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'


##################################
############ add Bal ###########
##################################


# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/plain/train_addition_bal.py \
#     --wandb_run_name="plain_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_plain_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='plain' --operator='-' \
# --dataset='bal' --out_dir='out2/subtraction_plain' \
# --start="FILE:data/bal/test_10000.txt" --device='cuda:0'



# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/dollar_reverse/train_addition_dollar_reverse.py \
#     --wandb_run_name="reverse_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_reverse_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='reverse' --operator='-' --reverse_c=True \
# --dataset='bal' --out_dir='out2/subtraction_reverse' \
# --start="FILE:data/bal/test_10000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/algorithmic_reasoning/train_addition_ar.py \
#     --wandb_run_name="ar_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_ar_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='-' --algo_reason=True \
# --dataset='bal' --out_dir='out2/subtraction_ar' \
# --start="FILE:data/bal/test_1000.txt" --device='cuda:0'

##################################
############ Simple AR ###########
##################################


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/ar_simple/train_addition.py \
#     --wandb_run_name="ar_simple_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_ar_simple_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='-' --algo_reason=True \
# --simple=True --random_A=False --random_C=False \
# --dataset='bal' --out_dir='out2/subtraction/ar_simple' \
# --start="FILE:data/bal/test_3digit_1000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/ar_simple/train_addition_randomA.py \
#     --wandb_run_name="ar_simple_randomA_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

for num_train in {5000,}; do
    echo $num_train
    python train.py config2/subtraction/ar_simple/train_addition_randomA.py \
    --wandb_run_name="ar_simple_randomA_${num_train}" \
    --train_data_path="train_3digit_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt"
done

python sample_addition.py \
--wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_ar_simple_randomA_Zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='algo_reasoning' --operator='-' --algo_reason=True \
--simple=True --random_A=True --random_C=False \
--dataset='bal' --out_dir='out2/subtraction/ar_simple_randomA' \
--start="FILE:data/bal/test_3digit_1000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/ar_simple/train_addition_randomC.py \
#     --wandb_run_name="ar_simple_randomC_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_ar_simple_randomC_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='-' --algo_reason=True \
# --simple=True --random_A=False --random_C=True \
# --dataset='bal' --out_dir='out2/subtraction/ar_simple_randomC' \
# --start="FILE:data/bal/test_3digit_1000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config2/subtraction/ar_simple/train_addition_randomboth.py \
#     --wandb_run_name="ar_simple_randomboth_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='subtraction' --wandb_run_name='eval_ar_simple_randomboth_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='-' --algo_reason=True \
# --simple=True --random_A=True --random_C=True \
# --dataset='bal' --out_dir='out2/subtraction/ar_simple_randomboth' \
# --start="FILE:data/bal/test_3digit_1000.txt" --device='cuda:0'



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
