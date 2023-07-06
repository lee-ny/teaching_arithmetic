# training

### Task 3. WITH REVERSE

# for lr in {1e-3,1e-4,1e-5,1e-6}; do # (MIXED) best: 1e-4
#     echo $lr
#     python train.py config/train_shakespeare_addition_reverse_large.py \
#     --wandb_run_name="large_separate_0.5_lr_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done


# for lr in {1e-3,1e-4,1e-5,1e-6}; do # (MIXED) best: 1e-3
#     echo $lr
#     python train.py config/train_shakespeare_addition_reverse2_large.py \
#     --wandb_run_name="large_mixed_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done

# # varying data_ratio
# for data_ratio in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}; do
#     echo $data_ratio
#     python train.py config/train_shakespeare_addition_reverse_large.py \
#     --wandb_run_name="large_separate_lr1e-4_${data_ratio}" \
#     --ckpt_path_name="ckpt_dr${data_ratio}.pt" \
#     --learning_rate=1e-4 --data_ratio=$data_ratio --device='cuda:0'
# done

# varying number of train samples in addition train set
# for num_train in {13000,26000,39000,52000,65000,78000,91000,104000,117000}; do
#     echo $num_train
#     python train.py config/train_shakespeare_addition_reverse_large.py \
#     --wandb_run_name="large_separate_lr1e-4_0.5_${num_train}" \
#     --train_data_path2="train_addition_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt" \
#     --learning_rate=1e-4
# done



# python sample_addition.py --start="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --out_dir=out-jt-shakespeare-addition-reverse-large --plot_sample_acc=True --dataset='shakespeare_addition_reverse' \
# --reverse_c=True --wandb_project='jt-shakespeare-addition-reverse' --wandb_run_name='large_dataratio_vs_accuracy' --dtype=float16


# python sample_shakespeare.py --eval_text_data_path='data/shakespeare_addition_reverse/val_shakespeare.bin' \
# --out_dir=out-jt-shakespeare-addition-reverse-large --plot_sample_acc=True --dataset='shakespeare_addition_reverse' \
# --wandb_project='jt-shakespeare-addition-reverse' --wandb_run_name='large_dataratio_vs_ppl' --device='cuda:1' --dtype=float16


# python sample_addition.py --start="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --out_dir=out-jt-shakespeare-addition-reverse-large --plot_sample_acc=True --dataset='shakespeare_addition_reverse' \
# --reverse_c=True --wandb_project='jt-shakespeare-addition-reverse' --wandb_run_name='large_samplenum_vs_accuracy' --dtype=float16

# CUDA_VISITBLE_DEVICES=1 python sample_shakespeare.py --eval_text_data_path='data/shakespeare_addition_reverse/val_shakespeare.bin' \
# --out_dir=out-jt-shakespeare-addition-reverse-large --plot_sample_acc=True --dataset='shakespeare_addition_reverse' \
# --wandb_project='jt-shakespeare-addition-reverse' --wandb_run_name='large_samplenum_vs_ppl' --device='cuda' --dtype=float16


###########################################
###########################################
###########################################

### Task 3. Jointly train shakespeare and Addition (WITHOUT Reverse, with dollar)



# for lr in {1e-3,1e-4,1e-5,1e-6}; do ## best: ???? # TODO:
#     echo $lr
#     python train.py config/jt_shakespeare_addition/train_shakespeare_addition_dollar_mix_large.py \
#     --wandb_run_name="large_mixed_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done


# varying data_ratio
# for data_ratio in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}; do
#     echo $data_ratio
#     python train.py config/jt_shakespeare_addition/train_shakespeare_addition_dollar_large.py \
#     --wandb_run_name="large_${data_ratio}" \
#     --ckpt_path_name="ckpt_dr${data_ratio}.pt" \
#     --learning_rate=1e-4 --data_ratio=$data_ratio --device='cuda:0'
# done

# varying number of train samples in addition train set
# for num_train in {13000,26000,39000,52000,65000,78000,91000,104000,117000}; do
#     echo $num_train
#     python train.py config/jt_shakespeare_addition/train_shakespeare_addition_dollar_large.py \
#     --wandb_run_name="large_${num_train}" \
#     --train_data_path2="train_addition_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt" \
#     --learning_rate=1e-4
# done


python sample_addition.py --start=FILE:prompt_dollar/prompt_addition_pad_test_0.01.txt \
--out_dir='out/out-jt-shakespeare-addition-large' --plot_sample_acc=True --dataset='shakespeare_addition_dollar' \
--wandb_project='jt-shakespeare-addition' --wandb_run_name='large_dataratio_vs_accuracy' --select='dataratio'

python sample_shakespeare.py --eval_text_data_path='data/shakespeare_addition_dollar/val_shakespeare.bin' \
--out_dir='out/out-jt-shakespeare-addition-large' --plot_sample_acc=True --dataset='shakespeare_addition_dollar' \
--wandb_project='jt-shakespeare-addition' --wandb_run_name='large_dataratio_vs_ppl' --select='dataratio'

# python sample_addition.py --start=FILE:prompt_dollar/prompt_addition_pad_test_0.01.txt \
# --out_dir=out-jt-shakespeare-addition --plot_sample_acc=True --dataset='shakespeare_addition_dollar' \
# --wandb_project='jt-shakespeare-addition' --wandb_run_name='large_samplenum_vs_accuracy' --select='samplenum'

# python sample_shakespeare.py --eval_text_data_path='data/shakespeare_addition_dollar/val_shakespeare.bin' \
# --out_dir=out-jt-shakespeare-addition --plot_sample_acc=True --dataset='shakespeare_addition_dollar' \
# --wandb_project='jt-shakespeare-addition' --wandb_run_name='large_samplenum_vs_ppl' --select='samplenum'

# for lr in {1e-3,1e-4,1e-5,1e-6}; do ## best: ???? # TODO:
#     echo $lr
#     python train.py config/jt_shakespeare_addition/train_shakespeare_addition_dollar_large.py \
#     --wandb_run_name="large_separate_0.5_lr_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done




###########################################
###########################################
###########################################



# python 0307_train_decay_dr.py config/train_shakespeare_addition_reverse.py \
# --wandb_run_name="separate_lr1e-4_decay_dr_v2" \
# --ckpt_path_name="ckpt_decay_dr_v2.pt" \
# --learning_rate=1e-4 --device='cuda:1' \
# --batch_size=128 --eval_interval=500 --max_iters=40000 --lr_decay_iters=40000

# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-reverse --ckpt_path_name=ckpt_decay_dr_acc.pt --evaluate=False --dataset=shakespeare_addition_reverse --num_samples=1




# python train.py config/finetune_shakespeare_addition_pad2.py \
#     --wandb_run_name="mini-gpt-padded_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"

# for num_train in {13000,26000,39000,52000,65000,78000,91000,104000,117000}; do
#     echo $num_train
#     python train.py config/train_addition_pad.py \
#     --wandb_run_name="mini-gpt-padded_${num_train}" \
#     --train_data_path="train_${num_train}.bin" --val_data_path="val_${num_train}.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


## Task2: finetune on addition for pretrained on shakespeare - best: 1e-4
# for lr in {1e-3,1e-4,1e-5,1e-6,1e-7,1e-8}; do
#     echo $lr
#     python train.py config/finetune_shakespeare_addition_pad2.py \
#     --wandb_run_name="mini-gpt-padded_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done

# for num_train in {91000,104000,117000,78000}; do
#     echo $num_train
#     python train.py config/finetune_shakespeare_addition_pad2.py \
#     --wandb_run_name="mini-gpt-padded_${num_train}" \
#     --learning_rate=1e-4 \
#     --train_data_path="train_addition_${num_train}.bin" --val_data_path="val_addition.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done


# Task3: jointly train addition / shakespeare - best: 1e-3
# for lr in {1e-3,1e-4,1e-5,1e-6,1e-7,1e-8}; do
#     echo $lr
#     python train.py config/train_shakespeare_addition_pad.py \
#     --wandb_run_name="mini-gpt-padded_${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done

# python train.py config/train_shakespeare_addition_pad.py --wandb_log=False --learning_rate=1e-3 --ckpt_path_name="aaa.pt" --eval_text=False
# {26000,52000,78000,104000}
# for num_train in {13000,39000,65000,91000,117000}; do
#     echo $num_train
#     python train.py config/train_shakespeare_addition_pad.py \
#     --wandb_run_name="mini-gpt-padded_${num_train}" \
#     --learning_rate=1e-3 \
#     --train_data_path="train_addition_${num_train}.bin" --val_data_path="val_all.bin" \
#     --ckpt_path_name="ckpt_${num_train}.pt"
# done

##############################################
##############################################

### testing et tu brute
# python sample.py --start="FILE:prompt/prompt_shakespeare.txt" --out_dir=out-shakespeare-char
# python sample.py --start="FILE:prompt/prompt_shakespeare.txt" --out_dir=out-shakespeare-char --evaluate=True


# python sample.py  --out_dir=out-ft-shakespeare-addition-char-pad --ckpt_path_name=ckpt_26000_final.pt --dataset=shakespeare_addition_char --evaluate=False --num_samples=1 --max_new_tokens=40 --start='et tu brute' 

# python sample.py  --out_dir=out-jt-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-3.pt --dataset=shakespeare_addition_char --evaluate=False --num_samples=1 --max_new_tokens=40 --start='et tu brute' 


### testing Addition
# python sample_addition.py --start="FILE:prompt/prompt_addition_test_0.001.txt" --out_dir=out-addition-char
# python sample_addition.py --start="FILE:prompt/prompt_addition_pad_test_0.01.txt" --out_dir=out-addition-char-pad --plot_sample_acc=True --wandb_run_name='num_train-mini-gpt-padded_0.01'

# python sample_addition.py --device='cuda:1' --wandb_log=False --start="FILE:prompt/prompt_addition_pad_test_0.0001.txt" --out_dir=out-ft-shakespeare-addition-char-pad --plot_sample_acc=True --wandb_run_name='num_train-mini-gpt-padded_0.0001' --ckpt_path_name='ckpt_1e-3.pt'

# python sample_addition.py --start="FILE:prompt/prompt_addition_pad_test_0.01.txt" \
# --out_dir=out-ft-shakespeare-addition-char-pad --plot_sample_acc=True --dataset='shakespeare_addition_char' \
# --wandb_project='ft-shakespeare-addition-char' --wandb_run_name='num_train-mini-gpt-padded_0.01'



### evaluating saved models
# python evaluate_models.py --start="FILE:prompt/prompt_addition_pad_test_0.001.txt" --out_dir=out-ft-shakespeare-addition-char-pad --plot_sample_acc=True --analyze=True --wandb_project='ft-shakespeare-addition-char' --wandb_run_name='num_train-mini-gpt-padded_0.001'
# python evaluate_models.py --start="FILE:prompt/prompt_addition_pad_test_0.001.txt" --out_dir=out-jt-shakespeare-addition-char-pad --plot_sample_acc=True --analyze=True --wandb_project='jt-shakespeare-addition-char' --wandb_run_name='num_train-mini-gpt-padded_0.001'

# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-3.pt --evaluate=False --dataset=shakespeare_addition_char
# python sample.py --start="et tu brute" --out_dir=out-ft-addition-shakespeare-char-pad --ckpt_path_name=ckpt_1e-3_final.pt --evaluate=False --dataset=shakespeare_addition_char
# python sample.py --start="et tu brute" --out_dir=out-ft-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-3_final.pt --evaluate=False --dataset=shakespeare_addition_char --num_samples=1
# works for 1e-3~1e-5 only

# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-reverse --ckpt_path_name=ckpt_dr0.9_final.pt --evaluate=False --dataset=shakespeare_addition_char --num_samples=1


#### Joint
## joint (vanilla) - addition only works for 1e-3
# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-char-pad --ckpt_path_name=ckpt_1e-3_final.pt --evaluate=False --dataset=shakespeare_addition_char
# python evaluate_additions.py --wandb_log=False --device='cuda:0' \
# --reverse_c=False --zero_pad=False --dataset='shakespeare_addition_char' --out_dir='out-jt-shakespeare-addition-char-pad' \
# --ckpt_path_name=ckpt_1e-3_final.pt --max_new_tokens=4 \
# --prompt_overall="FILE:prompt/prompt_addition_pad_test_0.01.txt" \
# --prompt1="FILE:prompt/prompt_addition_pad_test_1.txt" \
# --prompt2="FILE:prompt/prompt_addition_pad_test_2_0.1.txt"


## joint (reversed) - addition only works for 1e-3, 1e-4
# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-reverse-mixed --ckpt_path_name=ckpt_1e-3_final.pt --evaluate=False --dataset=shakespeare_addition_reverse --num_samples=1
# python evaluate_additions.py --wandb_log=False --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='shakespeare_addition_reverse' --out_dir='out-jt-shakespeare-addition-reverse-mixed' \
# --ckpt_path_name=ckpt_1e-3_final.pt --max_new_tokens=4 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar_reverse2/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt_dollar_reverse2/prompt_addition_test_2_0.1.txt"

## joint (reversed & mixed) - addition only works for 1e-3
# python sample.py --start="et tu brute" --out_dir=out-jt-shakespeare-addition-reverse --ckpt_path_name=ckpt_1e-3_final.pt --evaluate=False --dataset=shakespeare_addition_reverse --num_samples=1
# python evaluate_additions.py --wandb_log=False --device='cuda:0' \
# --reverse_c=True --zero_pad=False --dataset='shakespeare_addition_reverse' --out_dir='out-jt-shakespeare-addition-reverse' \
# --ckpt_path_name=ckpt_1e-3_final.pt --max_new_tokens=4 \
# --prompt_overall="FILE:prompt_dollar_reverse2/prompt_addition_test_0.01.txt" \
# --prompt1="FILE:prompt_dollar_reverse2/prompt_addition_test_1.txt" \
# --prompt2="FILE:prompt_dollar_reverse2/prompt_addition_test_2_0.1.txt"





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


# python sample.py --start="814+151=" --out_dir=0301-out-addition-dollar-reverse-curr-bal2 --ckpt_path_name=ckpt_final.pt --evaluate=False --dataset=addition_dollar_reverse_curr_bal2 --num_samples=1 --max_new_tokens=8
