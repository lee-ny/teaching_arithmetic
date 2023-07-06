
# python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py

# ar_size=3000
# add_size=10000
# CUDA_VISIBLE_DEVICES=0 python train_mixed.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
# --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="ar${ar_size}_add${add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" --dtype='float16'

# ar_size=3000
# add_size=10000

# for add_size in {2000,5000,20000,40000}; do ## best: ???
#     CUDA_VISIBLE_DEVICES=2 python train_mixed.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
#     --device='cuda' --dtype='float16'
# done

# # ar_size=3000
# # add_size=10000

# for ar_size in {500,1000,2000,3000,5000}; do ## best: ???
#     python train_mixed.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
#     --device='cuda' --dtype='float16'
# done


### dropout
# ar_size=3000
# add_size=10000
# python train_mixed.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar_dropout.py \
# --out_dir='out-gpt-dropout' --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="dropout0_2_ar${ar_size}_add${add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
# --device='cuda:1'

## dropout + no flash attention
# ar_size=3000
# add_size=10000
# python train_mixed.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar_dropout.py \
# --out_dir='out-gpt-dropout-noflash' --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="noflash-dropout0_2_ar${ar_size}_add${add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
# --device='cuda:1' --use_flash=False


## idunn with dtype=float16 instead of bfloat16
# ar_size=3000
# add_size=40000

# python train_mixed_debug.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
# --out_dir='out-gpt/debug2_float16' --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="float16_ar${ar_size}_add$add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
# --device='cuda:0' --dtype='float16'


## debuggin idunn failure case:
# add_size=40000
# ar_size=3000
# python train_mixed_debug.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
# --out_dir='out-gpt/debug_torchnew' --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="torchnew_debug_ar${ar_size}_add${add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin"
# --device='cuda:1'


# add_size=40000
# ar_size=3000
# python train_mixed_debug.py config/gpt2/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
# --out_dir='out-gpt/debug_torchnew_float16' --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
# --wandb_run_name="float16_torchnew_debug_ar${ar_size}_add${add_size}" \
# --train_data_path="train_all_ar${ar_size}_add${add_size}.bin"
# --device='cuda:1' --dtype='float16'


###################################################
######## evaluating num_samples (zeroshot)#########
###################################################

# ar_size=3000
# add_size=10000
# lr=1e-3

# TODO:
# # evaluate different numbers of addition examples
# CUDA_VISIBLE_DEVICES=1 python sample_addition.py \
# --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_ar3000_zeroshot" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='mixed' \
# --start="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --num_ar=3000 --dtype=float16

# CUDA_VISIBLE_DEVICES=2 python sample_addition.py \
# --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_add10000_zeroshot" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='mixed' --algo_reason=True \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --num_add=10000 --dtype=float16

# ###################################################
# ######## evaluating few-shot prompts #############
# ###################################################

# # evaluate different numbers of ar examples
# for shots in {1,2,3}; do
#     CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
#     --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_add10000_${shots}shot" \
#     --algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
#     --plot_sample_acc=True --select='fewshot' \
#     --start="FILE:data/algo_reasoning/few_shot_prompts/prompt_addition_${shots}shot_test_0.0001_1.txt" \
#     --fewshot=True --num_add=10000 --dtype=float16
# done

# # # # evaluate different numbers of addition examples
# for shots in {1,2,3}; do
#     CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
#     --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_ar3000_${shots}shot" \
#     --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
#     --plot_sample_acc=True --select='fewshot' \
#     --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_${shots}shot_test_0.001_1.txt" \
#     --fewshot=True --num_ar=3000 --dtype=float16 
# done


# ###################################################
# ########### evaluating word prompts ###############
# ###################################################

# evaluate different numbers of ar examples
# CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_add10000" \
# --algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='fewshot' \
# --start="FILE:data/algo_reasoning/word_prompts/prompt_addition_0.0001_1.txt" \
# --fewshot=True --num_add=10000 --dtype=float16

# # # # evaluate different numbers of addition examples
# CUDA_VISIBLE_DEVICES=3 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_ar3000" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='fewshot' \
# --start="FILE:data/addition_bal/word_prompts/prompt_addition_0.01_1.txt" \
# --fewshot=True --num_ar=3000 --dtype=float16 

#######
# # # # evaluate multiple sets for each prompt set
#######

# # # evaluate different numbers of ar examples
CUDA_VISIBLE_DEVICES=3 python sample_addition_fewshot.py \
--wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_add10000_multiple" \
--algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
--plot_sample_acc=True --select='fewshot' --multiple_set_per_prompt=True \
--start="FILE:data/algo_reasoning/word_prompts/prompt_addition_0.0001_1_1.txt" \
--fewshot=True --num_add=10000 --dtype=float16

# # # # # evaluate different numbers of addition examples
# CUDA_VISIBLE_DEVICES=3 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='gpt2-jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_ar3000_multiple" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out-gpt/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='fewshot' --multiple_set_per_prompt=True \
# --start="FILE:data/addition_bal/word_prompts/prompt_addition_0.001_1_1.txt" \
# --fewshot=True --num_ar=3000 --dtype=float16 




###################################################
######## evaluating text generation #############
###################################################

# python sample.py --start="et tu brute" --out_dir=out-gpt/shakespeare_add_ar_mixed --dataset=shakespeare_add_ar_mixed \
# --ckpt_path_name="ckpt_ar500_add10000.pt" \
# --evaluate=False --num_samples=1


# python sample.py --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --ckpt_path_name='ckpt_ar3000_add20000_lr1e-3_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
# --start="165+182=347
# 986+184=1140
# 198+864="


# python sample.py --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --ckpt_path_name='ckpt_ar3000_add2000_lr1e-3_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
#  --dtype='float16' --start="165+182=347
# 986+184=1140
# 198+864="
# outputs: 198+864=1551

# python sample.py --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --ckpt_path_name='ckpt_ar3000_add2000_lr1e-3_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
#  --dtype='float16' --start="165+182=347
# 986+184=1140
# 198+864="
# # outputs: 198+864=1055


# python sample.py --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --ckpt_path_name='ckpt_ar3000_add5000_lr1e-3_final.pt' --evaluate=False --num_samples=1 --max_new_tokens=4 \
#  --dtype='float16' --start="165+182=347
# 986+184=1140
# 198+864=1055
# 984+382="