
for num_train in {10000,}; do
    echo $num_train
    CUDA_VISIBLE_DEVICES=2 python train.py config/addition_large_vocab/train_addition_bal.py --device='cuda' \
    --wandb_run_name="addition_${num_train}" \
    --train_data_path="train_${num_train}.bin" --val_data_path="val.bin" \
    --ckpt_path_name="ckpt_${num_train}.pt" --dtype=float16
done


CUDA_VISIBLE_DEVICES=1 python sample_addition.py \
--wandb_log=True --wandb_project='addition-large-vocab' --wandb_run_name='zeroshot' \
--plot_sample_acc=True --select='samplenum' \
--dataset='addition_large_vocab' --out_dir='out/out-addition-large-vocab' \
--start="FILE:data/addition_bal/prompt_addition_test_0.01.txt" --device='cuda' --dtype=float16


# # # # few-shot! evaluate different numbers of addition examples
for shots in {1,2,3}; do
    echo $shots
    CUDA_VISIBLE_DEVICES=0 python sample_addition_fewshot.py \
    --wandb_log=True --wandb_project='addition-large-vocab' --wandb_run_name="add_${shots}shot" \
    --dataset='addition_large_vocab' --out_dir='out/out-addition-large-vocab' \
    --plot_sample_acc=True --select='samplenum' \
    --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_${shots}shot_test_0.01_1.txt" \
    --fewshot=True --device='cuda' --dtype=float16
done


##### word prompt
# CUDA_VISIBLE_DEVICES=0 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='addition-large-vocab' --wandb_run_name="eval_word_prompt" \
# --dataset='addition_large_vocab' --out_dir='out/out-addition-large-vocab' \
# --plot_sample_acc=True --select='samplenum' \
# --start="FILE:data/addition_bal/word_prompts/prompt_addition_0.01_1.txt" \
# --fewshot=True --dtype=float16 

##### word prompt multiple
CUDA_VISIBLE_DEVICES=3 python sample_addition_fewshot.py \
--wandb_log=True --wandb_project='addition-large-vocab' --wandb_run_name="eval_word_prompt_multiple" \
--dataset='addition_large_vocab'  --out_dir='out/out-addition-large-vocab' \
--plot_sample_acc=True --select='samplenum' --multiple_set_per_prompt=True \
--start="FILE:data/addition_bal/word_prompts/prompt_addition_0.01_1_1.txt" \
--fewshot=True --dtype=float16 


## evaluation on noisy prompt
CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
--wandb_log=True --wandb_project='addition-large-vocab' --wandb_run_name="add_noisyprompt" \
--dataset='addition_large_vocab'  --out_dir='out/out-addition-large-vocab' \
--plot_sample_acc=True --select='samplenum' \
--start="FILE:data/addition_bal/few_shot_noisy_prompts/prompt_addition_3shot_test_0.01_1.txt" \
--fewshot=True --device='cuda' --dtype=float16

