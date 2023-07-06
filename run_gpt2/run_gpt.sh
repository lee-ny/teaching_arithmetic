# training
### run on GPT-2
### shakespeare
##### best: 5e-5 {5e-5,1e-5,5e-6,1e-6};
# for lr in {5e-5,}; do
#     echo $lr
#     python train.py config/finetune_shakespeare.py \
#     --wandb_run_name="ft-gpt2-lr${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done

### addition 
##### best: 1e-4
# for lr in {1e-4,}; do
#     echo $lr
#     python train.py config/finetune_addition.py \
#     --wandb_run_name="ft-gpt2-lr${lr}" \
#     --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
# done

### all
##### best: 

for lr in {5e-5,1e-5,5e-6,1e-6}; do
    echo $lr
    python train.py config/finetune_shakespeare_addition.py \
    --wandb_run_name="ft-gpt2-lr${lr}" \
    --learning_rate=$lr --ckpt_path_name="ckpt_${lr}.pt"
done

# testing
# "FILE:prompt/prompt_addition_pad_test_0.0001.txt"
# python sample_addition.py --out_dir=out-addition --ckpt_path_name=ckpt_1e-4.pt --plot_sample_acc=False --dataset=shakespeare_addition --start="913+894="

