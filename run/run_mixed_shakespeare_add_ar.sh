
# python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py

# CUDA_VISIBLE_DEVICES=1

# ar_size=3000
# add_size=10000

# # for lr in {1e-3,1e-4,1e-2}; do ## best: ???
# for lr in {1e-3,}; do ## best: ???
#     echo $lr
#     python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --learning_rate=$lr --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}_lr${lr}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}_lr${lr}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin"
# done

# lr=1e-3
# for ar_size in {500,1000,2000,3000,5000}; do ## best: ???
#     echo $lr
#     CUDA_VISIBLE_DEVICES=0 python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin" \
#     --device='cuda' --dtype='float16'
# done


# ar_size=3000
# add_size=10000
# lr=1e-3

# for add_size in {2000,5000,10000,20000,40000}; do ## best: ???
#     echo $lr
#     python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --learning_rate=$lr --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}_lr${lr}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}_lr${lr}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin"
#     --device='cuda:1'
# done


# for add_size in {2000,5000,20000,40000}; do ## best: ???
#     echo $lr
#     python train_mixed.py config/shakespeare_add_ar_mixed/train_shakespeare_add_ar.py \
#     --learning_rate=$lr --ckpt_path_name="ckpt_ar${ar_size}_add${add_size}_lr${lr}.pt" \
#     --wandb_run_name="ar${ar_size}_add${add_size}_lr${lr}" \
#     --train_data_path="train_all_ar${ar_size}_add${add_size}.bin"
#     --device='cuda:1'
# done



###################################################
######## evaluating num_samples (zeroshot)#########
###################################################

# ar_size=3000
# add_size=10000
# lr=1e-3

# # # evaluate different numbers of addition examples
# CUDA_VISIBLE_DEVICES=0 python sample_addition.py \
# --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_ar3000_zeroshot" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='mixed' \
# --start="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --num_ar=3000 --dtype=float16

# CUDA_VISIBLE_DEVICES=3 python sample_addition.py \
# --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_add10000_zeroshot" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='mixed' --algo_reason=True \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --num_add=10000 --dtype=float16


##### USE THIS TO EVALUTE A SINGLE MODEL -> Specify out_dir and ckpt_path_name
# CUDA_VISIBLE_DEVICES=3 python sample_addition.py \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' --ckpt_path_name="ckpt_ar3000_add10000_lr1e-3_final.pt" \
# --plot_sample_acc=False --select='mixed' \
# --start="FILE:data/addition_bal/prompt_addition_test_0.01.txt" \
# --num_ar=3000 --dtype=float16 --wandb_log=False

# CUDA_VISIBLE_DEVICES=1 python sample_addition.py \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' --ckpt_path_name="ckpt_ar3000_add10000_lr1e-3_final.pt" \
# --plot_sample_acc=False --select='mixed' --algo_reason=True \
# --start="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --num_add=10000 --dtype=float16


###################################################
######## evaluating few-shot prompts #############
###################################################

# # evaluate different numbers of ar examples
for shots in {1,2,3}; do
    CUDA_VISIBLE_DEVICES=0 python sample_addition_fewshot.py \
    --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_add10000_${shots}shot" \
    --algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
    --plot_sample_acc=True --select='fewshot' \
    --start="FILE:data/algo_reasoning/few_shot_prompts/prompt_addition_${shots}shot_test_0.0001_1.txt" \
    --fewshot=True --num_add=10000 --dtype=float16
done

# # # evaluate different numbers of addition examples
for shots in {1,2,3}; do
    CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
    --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_ar3000_${shots}shot" \
    --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
    --plot_sample_acc=True --select='fewshot' \
    --start="FILE:data/addition_bal/few_shot_prompts/prompt_addition_${shots}shot_test_0.001_1.txt" \
    --fewshot=True --num_ar=3000 --dtype=float16 
done

# ###################################################
# ########### evaluating word prompts ###############
# ###################################################


# # evaluate different numbers of ar examples
# CUDA_VISIBLE_DEVICES=2 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_add10000" \
# --algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='fewshot' \
# --start="FILE:data/algo_reasoning/word_prompts/prompt_addition_0.0001_1.txt" \
# --fewshot=True --num_add=10000 --dtype=float16

# # # # evaluate different numbers of addition examples
# CUDA_VISIBLE_DEVICES=1 python sample_addition_fewshot.py \
# --wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_ar3000" \
# --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
# --plot_sample_acc=True --select='fewshot' \
# --start="FILE:data/addition_bal/word_prompts/prompt_addition_0.01_1.txt" \
# --fewshot=True --num_ar=3000 --dtype=float16 

#######
# # # # evaluate multiple sets for each prompt set
#######
# evaluate different numbers of ar examples
CUDA_VISIBLE_DEVICES=0 python sample_addition_fewshot.py \
--wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_add10000_multiple" \
--algo_reason=True --dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
--plot_sample_acc=True --select='fewshot' --multiple_set_per_prompt=True \
--start="FILE:data/algo_reasoning/word_prompts/prompt_addition_0.0001_1_1.txt" \
--fewshot=True --num_add=10000 --dtype=float16

# # # # evaluate different numbers of addition examples
CUDA_VISIBLE_DEVICES=3 python sample_addition_fewshot.py \
--wandb_log=True --wandb_project='jt-shakespeare-add-ar-mixed' --wandb_run_name="eval_word_prompt_ar3000_multiple" \
--dataset='shakespeare_add_ar_mixed' --out_dir='out/shakespeare_add_ar_mixed' \
--plot_sample_acc=True --select='fewshot' --multiple_set_per_prompt=True \
--start="FILE:data/addition_bal/word_prompts/prompt_addition_0.001_1_1.txt" \
--fewshot=True --num_ar=3000 --dtype=float16 



###################################################
######## evaluating text generation #############
###################################################

# python sample.py --start="et tu brute" --out_dir=out-gpt/shakespeare_add_ar_mixed --dataset=shakespeare_add_ar_mixed \
# --ckpt_path_name="ckpt_ar1000_add10000.pt" \
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