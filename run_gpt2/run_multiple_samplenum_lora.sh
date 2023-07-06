# # python train.py config/algo_reasoning/train_addition.py --wandb_run_name='lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'

# ######################################
# ############ GPT Tokenizer ###########
# ##########   from scratch   ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_${num_train}" \
#     --train_data_path="train_3digit_${num_train}.txt" \
#     --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
# done


# ######################################
# ############ GPT Tokenizer ###########
# ##########   from scratch   ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_${num_train}_lr1e-4" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt"
# done

# #####################################
# ########### GPT Tokenizer ###########
# ######### from pretrained  ##########
# #####################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_${num_train}_lr1e-4" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt" --device='cuda:1'
# done


# ####################################
# ########## GPT Tokenizer ###########
# ########   from scratch   ##########
# ####################################



# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done



######################################
############ GPT Tokenizer ###########
##########   from scratch   ##########
######################################


for num_train in {10000,}; do
    echo $num_train
    python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_space.py \
    --wandb_run_name="lora-plain_gpt_tokenizer_space_${num_train}" \
    --train_data_path="train_3digit_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt"
done

######################################
############ GPT Tokenizer ###########
########## from pretrained  ##########
######################################


for num_train in {10000,}; do
    echo $num_train
    python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained_space.py \
    --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_space_${num_train}" \
    --train_data_path="train_3digit_${num_train}.txt" \
    --ckpt_path_name="ckpt_${num_train}.pt" --device='cuda:0'
done


######################################
############ GPT Tokenizer ###########
##########   from scratch   ##########
######################################


for num_train in {10000,}; do
    echo $num_train
    python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_space.py \
    --wandb_run_name="lora-plain_gpt_tokenizer_space_${num_train}_lr1e-4" \
    --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
    --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt"
done

# #####################################
# ########### GPT Tokenizer ###########
# ######### from pretrained  ##########
# #####################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained_space.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_space_${num_train}_lr1e-4" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=1e-4 \
#     --ckpt_path_name="ckpt_${num_train}_lr1e-4.pt" --device='cuda:1'
# done


# ####################################
# ########## GPT Tokenizer ###########
# ########   from scratch   ##########
# ####################################



# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_space.py \
#     --wandb_run_name="lora-plain_gpt_tokenizer_space_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done

# ######################################
# ############ GPT Tokenizer ###########
# ########## from pretrained  ##########
# ######################################


# for num_train in {10000,}; do
#     echo $num_train
#     python train.py config_gpt2/addition/plain_lora/train_addition_bal_gpt_tokenizer_from_pretrained_spacepy \
#     --wandb_run_name="lora-plain_gpt_tokenizer_from_pretrained_space_${num_train}_lr3e-5" \
#     --train_data_path="train_3digit_${num_train}.txt" --learning_rate=3e-5 \
#     --ckpt_path_name="ckpt_${num_train}_lr3e-5.pt" --device='cuda:1'
# done
