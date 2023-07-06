# # python train.py config/algo_reasoning/train_addition.py --wandb_run_name='lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'


# ##################################
# ############ add Bal ###########
# ##################################

# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal.py \
#     --wandb_run_name="plain_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='addition-gpt' --wandbs_run_name='eval_plain_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='plain' --operator='+' \
# --dataset='bal' --out_dir='out2-gpt/addition_plain' \
# --start="FILE:data/bal/test_10000.txt" --device='cuda:0'



# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config_gpt2/addition/reverse/train_addition_dollar_reverse.py \
#     --wandb_run_name="reverse_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='addition-gpt' --wandb_run_name='eval_reverse_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='reverse' --operator='+' --reverse_c=True \
# --dataset='bal' --out_dir='out2-gpt/addition_reverse' \
# --start="FILE:data/bal/test_10000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config_gpt2/addition/ar/train_addition_ar.py \
#     --wandb_run_name="ar_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='additin-gpt' --wandb_run_name='eval_ar_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True \
# --dataset='bal' --out_dir='out2-gpt/addition_ar' \
# --start="FILE:data/bal/test_1000.txt" --device='cuda:0'



# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal2.py \
#     --wandb_run_name="plain_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

python sample_addition.py \
--wandb_log=True --wandb_project='addition-gpt' --wandb_run_name='eval_plain_Zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='plain' --operator='+' \
--dataset='bal' --out_dir='out2-gpt/addition_plain' \
--start="FILE:data/bal/test_10000.txt" --device='cuda:0'



for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
    echo $num_train
    python train.py config_gpt2/addition/reverse/train_addition_dollar_reverse2.py \
    --wandb_run_name="reverse_${num_train}" \
    --train_data_path="train_3digit_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
done

python sample_addition.py \
--wandb_log=True --wandb_project='addition-gpt' --wandb_run_name='eval_reverse_Zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True \
--dataset='bal' --out_dir='out2-gpt/addition_reverse' \
--start="FILE:data/bal/test_10000.txt" --device='cuda:0'


# for num_train in {1000,2000,3000,4000,5000,10000,20000}; do
#     echo $num_train
#     python train.py config_gpt2/addition/ar/train_addition_ar2.py \
#     --wandb_run_name="ar_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:1'
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='additin-gpt' --wandb_run_name='eval_ar_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True \
# --dataset='bal' --out_dir='out2-gpt/addition_ar' \
# --start="FILE:data/bal/test_1000.txt" --device='cuda:1'



######################################
############ GPT Tokenizer ###########
##########   from scratch   ##########
######################################



# for num_train in {10000,}; do
# ##################################
# ############ add Bal ###########
# ##################################
# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer.py \
#     --wandb_run_name="plain_gpt_tokenizer_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-3.pt"
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='addition-gpt' --wandbs_run_name='eval_plain_gpt_tokenizer_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='plain' --operator='+' --tokenizer='gpt2' \
# --dataset='bal' --out_dir='out2-gpt/addition_plain' \
# --start="data/bal/test_3digit_10000.txt" --device='cuda:0'

# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal2.py \
#     --wandb_run_name="plain_${num_train}_lr1e-4" \
#     --learning_rate=1e-4 \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt" --device='cuda:1'
# done


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/reverse/train_addition_dollar_reverse_gpt_tokenizer.py \
#     --wandb_run_name="reverse_gpt_tokenizer_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
# done

# python sample_addition.py \
# --wandb_log=True --wandb_project='addition-gpt' --wandb_run_name='eval_reverse_gpt_tokenizer_Zeroshot' \
# --plot_sample_acc=True --select='samplenum' \
# --data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --tokenizer='gpt2' \
# --dataset='bal' --out_dir='out2-gpt/addition_reverse' \
# --start="FILE:data/bal/test_3digit_10000.txt" --device='cuda:0'





# ######################################
# ############ GPT Tokenizer ###########
# ##########   from scratch   ##########
# ######################################



# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_space_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
# done


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer.py \
#     --wandb_run_name="plain_gpt_tokenizer_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:0'
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_from_pretrained_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_from_pretrained_space_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
# done


# ######################################
# ############ GPT Tokenizer ###########
# ##########   from scratch   ##########
# ######################################



# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_space_${num_train}_lr1e-4" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt"
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_from_pretrained_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_from_pretrained_space_${num_train}_lr1e-4" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt" --device='cuda:1'
# done


# ######################################
# ############ GPT Tokenizer ###########
# ##########   from scratch   ##########
# ######################################



# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_space_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain/train_addition_bal_gpt_tokenizer_from_pretrained_space.py \
#     --wandb_run_name="plain_gpt_tokenizer_from_pretrained_space_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done
