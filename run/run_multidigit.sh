
# python train.py config/addition_multidigit/train_addition_multidigit_ver1.py --num_digit=7


# num_digit=7
# num_sample=100000
# dataset='addition_multidigit_bal_ver1'
# # for lr in {1e-3,1e-4,1e-2}; do ## best: ???
# for lr in {1e-3,}; do ## best: ???
#     echo $lr
#     python train.py config/addition_multidigit/train_addition_multidigit_ver1.py \
#     --num_digit=7 --learning_rate=$lr --ckpt_path_name="ckpt_${num_digit}digit_lr${learning_rate}_num${num_sample}.pt" \
#     --wandb_run_name="ver1_${num_digit}digit_lr${lr}_${num_sample}" --dataset=$dataset \
#     --start="FILE:data/${dataset}/prompt_test_${num_digit}digit_100.txt" \
#     --train_data_path="train_${num_digit}_${num_sample}.bin" --val_data_path="val_${num_digit}_${num_sample}.bin" \
#     --device='cuda:1' 
# done
# python train.py config/addition_multidigit/train_addition_multidigit_ver2.py --num_digit=7

# python train.py config/addition_multidigit/train_addition_multidigit_ver1_2and4.py

# python train.py config/addition_multidigit/train_addition_multidigit_ver1_1and3.py

# dataset='addition_multidigit_bal_ver2'
# # for lr in {1e-3,1e-4,1e-2}; do ## best: ???
# for lr in {1e-3,}; do ## best: ???
#     echo $lr
#     python train.py config/addition_multidigit/train_addition_multidigit_ver2.py \
#     --num_digit=7 --learning_rate=$lr --ckpt_path_name='ckpt_${num_digit}digit_lr${learning_rate}_num${num_sample}.pt' \
#     --wandb_run_name='ver2_${num_digit}digit_lr${lr}_${num_sample}' \
#     --start="FILE:data/${dataset}/prompt_test_${num_digit}digit_100.txt" \
#     --train_data_path = f'train_${num_digit}_${num_sample}.bin' --val_data_path = f'val_${num_digit}_${num_sample}.bin' \
#     --device='cuda:1' 
# done


# python sample.py --out_dir=out/addition_multidigit_ver2 --dataset=addition_multidigit_bal_ver2 --evaluate=False \
# --ckpt_path_name=ckpt_7digit_lr_num100000_final.pt --num_samples=1 --max_new_tokens=11 --start='$98431194+65982316='

# python sample.py --out_dir=out/addition_multidigit_ver2 --dataset=addition_multidigit_bal_ver2 --evaluate=False \
# --ckpt_path_name=ckpt_7digit_lr_num100000_final.pt --num_samples=1 --max_new_tokens=11 --start='$9843119+6598231='

################################
######### Plain ################
################################

num_digit=4

# for num_sample in {1000,5000,10000,20000}; do
#     python train.py config2/multi_digit/plain/train_addition_bal.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
# done

# python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/plain/ckpt_10000_final.pt" \
# --data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
# --num_digit=4 --start="FILE:data/multi_digit/test_4digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_4digit' \
# --dtype='float16'


# for num_sample in {500,1000,5000,10000}; do
#     python train.py config2/multi_digit/reverse/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
# done

python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/reverse/ckpt_10000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_4digit' \
--dtype='float16'


# num_digit=4
# # running magnetes #4
# for num_sample in {500,1000,2000,5000}; do
#     CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/train_addition_ar2.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=20000 --lr_decay_iters=20000
# done

# eval
# CUDA_VISIBLE_DEVICES=3 python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/ar/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
# --num_digit=4 --start="FILE:data/multi_digit/test_4digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_4digit' \


# num_digit=4
# for num_sample in {250,500,1000,2000,5000}; do
#     CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/ar_simple/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=20000 --lr_decay_iters=20000 --dtype='float16'
# done


# CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/simple/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True  --simple=True --multi_digit=True --multi_model=True \
# --num_digit=4 --start="FILE:data/multi_digit/test_4digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_simple_4digit' \
# --dtype='float16'


num_digit=5

# for num_sample in {100000,10000,20000,50000}; do
#     CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/train_addition_bal.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
# done

# for num_sample in {500,1000,5000,10000,50000,100000}; do
#     CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
# done

# for num_sample in {500,1000,5000,100000,10000,50000,200000}; do
#     CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/train_addition_ar.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --device='cuda' --dtype='float16'
# done

# # for num_sample in {500,1000,5000,10000}; do
# #     python train.py config2/multi_digit/ar/train_addition_ar2.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=30000 --lr_decay_iters=30000
# done

# for num_sample in {500,1000,5000,10000}; do
#     python train.py config2/multi_digit/ar_simple/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=25000 --lr_decay_iters=25000
# done


num_digit=7

# for num_sample in {100000,10000,50000,200000}; do
#     CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/train_addition_bal.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16'
# done

# for num_sample in {500,1000,5000,100000,10000,50000,200000}; do
#     CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16'
# done


# # for num_sample in {500,1000,5000,100000,10000,50000,200000}; do
# #     CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/train_addition_ar.py \
# #     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
# #     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
# #     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
# #     --device='cuda' --dtype='float16'
# # done


# for num_sample in {500,1000,5000,10000}; do # TODO: running on ida
#     python train.py config2/multi_digit/ar/train_addition_ar2.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --device='cuda' --dtype='float16'
# done


# for num_sample in {500,1000,5000,10000}; do
#     python train.py config2/multi_digit/ar_simple/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=30000 --lr_decay_iters=30000 --device='cuda:1'
# done


# num_digit=10

# for num_sample in {5000,10000,50000,100000,500000}; do
#     CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/train_addition_bal.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16' --max_iters=25000 --lr_decay_iters=25000 
# done

# for num_sample in {1000,5000,10000,50000,100000}; do
#     CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
#     --device='cuda' --dtype='float16'
# done


# num_digit=10
# for num_sample in {500,1000,2000,3000,5000,10000}; do # TODO: Running on ida
#     python train.py config2/multi_digit/ar/train_addition_ar2.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=40000 --lr_decay_iters=40000 --device='cuda:0'
# done

# num_digit=10 #
# for num_sample in {1000,5000,10000,50000}; do 
#     CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/ar_simple/train_addition.py \
#     --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=35000 --lr_decay_iters=35000 --dtype='float16'
# done




# TODO: RUNNING ADDITIONAL EXPS


### Magnetes 2
num_digit=5
for num_sample in {1000,2000,5000,7000,10000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/reverse/train_addition.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16'
done

CUDA_VISIBLE_DEVICES=3 python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/reverse/ckpt_10000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=5 --start="FILE:data/multi_digit/test_5digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_5digit' \
--dtype='float16'


### Lee 1 {2000,3000}
num_digit=5
for num_sample in {500,10000}; do
    python train.py config2/multi_digit/ar/train_addition_ar2.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --max_iters=30000 --lr_decay_iters=30000
done

python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/ar/ckpt_10000_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
--num_digit=5 --start="FILE:data/multi_digit/test_5digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_5digit' \
--device='cuda:0'


### Magnetes 3
num_digit=7
for num_sample in {2000,5000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/reverse/train_addition.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16'
done

CUDA_VISIBLE_DEVICES=1 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/reverse/ckpt_10000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=7 --start="FILE:data/multi_digit/test_7digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_7digit' \
--dtype='float16'

### ida 1,3
num_digit=7
for num_sample in {3000,}; do
    python train.py config2/multi_digit/ar/train_addition_ar2.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda:1' --test_batch_size=64
done

python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/ar/ckpt_10000_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
--num_digit=7 --start="FILE:data/multi_digit/test_7digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_7digit' \
--device='cuda:1'


### Magnetes 1 (run_multidigit4.sh)
num_digit=7
for num_sample in {20000,}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/train_addition_bal.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16'
done

CUDA_VISIBLE_DEVICES=1 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/plain/ckpt_10000_final.pt" \
--data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
--num_digit=7 --start="FILE:data/multi_digit/test_7digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_7digit' \
--dtype='float16'


num_digit=10
for num_sample in {20000,}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/train_addition_bal.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16'
done

CUDA_VISIBLE_DEVICES=1 python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/plain/ckpt_10000_final.pt" \
--data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
--num_digit=10 --start="FILE:data/multi_digit/test_10digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_10digit' \
--dtype='float16'


# Lee 3
num_digit=10
for num_sample in {1000,5000,7000,10000,20000}; do
    python train.py config2/multi_digit/reverse/train_addition.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda:0'
done


python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/reverse/ckpt_10000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=10 --start="FILE:data/multi_digit/test_10digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_10digit' --device='cuda:0'


# Lee 4 -> Magnetes 2
num_digit=4
for num_sample in {3000,}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/reverse/train_addition.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda'
done

CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/reverse/ckpt_10000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_4digit' --device='cuda:0'


# Magnetes4 #TODO: No space right now. will run it later
num_digit=4
for num_sample in {10000,20000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/ar_simple/train_addition.py \
    --num_digit=$num_digit --out_dir="out2_multidigit/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --max_iters=20000 --lr_decay_iters=20000 --dtype='float16'
done


CUDA_VISIBLE_DEVICES=1 python test_addition.py --ckpt_path_name="out2_multidigit/digit_4/simple/ckpt_10000_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True  --simple=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_simple_4digit' \
--dtype='float16'




################################
######### Evaluation ###########
################################

# # 7 digit

# test on just 1-digit addition
# CUDA_VISIBLE_DEVICES=2 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/plain/ckpt_10000_final.pt" \
# --data_type='text' --data_format='plain' --operator='+' \
# --num_digit=7 --start="FILE:data/multi_digit/test_1digit_100.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_7digit_test1digit' \
# --dtype='float16' > out2_multidigit/digit_7/plain/eval_plain_7digit_test1digit.txt


# CUDA_VISIBLE_DEVICES=2 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/plain/ckpt_10000_final.pt" \
# --data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
# --num_digit=7 --start="FILE:data/multi_digit/test_7digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_7digit' \
# --dtype='float16'


# CUDA_VISIBLE_DEVICES=3 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/reverse/ckpt_10000_final.pt" \
# --data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
# --num_digit=7 --start="FILE:data/multi_digit/test_7digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_7digit' \
# --dtype='float16'


# python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/ar/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
# --num_digit=7 --start="FILE:data/multi_digit/test_7digit_100.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_7digit' \
# --device='cuda:0'


# CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit/digit_7/simple/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True  --simple=True --multi_digit=True --multi_model=True \
# --num_digit=7 --start="FILE:data/multi_digit/test_7digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_simple_7digit' \
# --device='cuda'



# # 5 digit

# CUDA_VISIBLE_DEVICES=2 python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/plain/ckpt_10000_final.pt" \
# --data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
# --num_digit=5 --start="FILE:data/multi_digit/test_5digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_5digit' \
# --dtype='float16'


# CUDA_VISIBLE_DEVICES=3 python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/reverse/ckpt_10000_final.pt" \
# --data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
# --num_digit=5 --start="FILE:data/multi_digit/test_5digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_5digit' \
# --dtype='float16'


# python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/ar/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
# --num_digit=5 --start="FILE:data/multi_digit/test_5digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_5digit' \
# --device='cuda:1'


# python test_addition.py --ckpt_path_name="out2_multidigit/digit_5/simple/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True  --simple=True --multi_digit=True --multi_model=True \
# --num_digit=5 --start="FILE:data/multi_digit/test_5digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_simple_5digit' \
# --device='cuda:1'


# # 10 digit

# CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/plain/ckpt_10000_final.pt" \
# --data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
# --num_digit=10 --start="FILE:data/multi_digit/test_10digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_plain_10digit' \
# --dtype='float16'


# CUDA_VISIBLE_DEVICES=2 python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/reverse/ckpt_10000_final.pt" \
# --data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
# --num_digit=10 --start="FILE:data/multi_digit/test_10digit_10000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_reverse_10digit' \
# --dtype='float16'

# TODO: ida
python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/ar/ckpt_10000_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
--num_digit=10 --start="FILE:data/multi_digit/test_10digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_ar_10digit' \
--device='cuda:1'

# CUDA_VISIBLE_DEVICES=1 python test_addition.py --ckpt_path_name="out2_multidigit/digit_10/simple/ckpt_10000_final.pt" \
# --data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True  --simple=True --multi_digit=True --multi_model=True \
# --num_digit=10 --start="FILE:data/multi_digit/test_10digit_1000.txt" \
# --wandb_log=True --wandb_project='addition_multidigit' --wandb_run_name='eval_simple_10digit' \
# --device='cuda'


##################################################################
############################# EXP 3. #############################
##################################################################


# Let's first train on 3-digit addition (note that the saved model must have same dtype with our finetune setting)
# so I'm retraining on Magnetes :( 
CUDA_VISIBLE_DEVICES=1 python train.py config2/addition/plain/train_addition_bal.py --dtype='float16'
CUDA_VISIBLE_DEVICES=3 python train.py config2/addition/ar/train_addition_ar.py --dtype='float16'
CUDA_VISIBLE_DEVICES=2 python train.py config2/addition/reverse/train_addition_bal.py --dtype='float16'
python train.py config2/subtraction/ar_simple/train_addition.py --dtype='float16' # this is done on aws

# first check best lr - let's first run for shorter time (5000 iters) -> seems like 5e-4!
num_digit=4
num_sample=10000
for lr in {1e-3,5e-4,1e-4,5e-5}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/finetune.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda:0' --max_iters=5000 --lr_decay_iters=5000 --dtype='float16'
done

# best for reverse: lr = 1e-4
num_digit=4
num_sample=5000
for lr in {1e-3,5e-4,1e-4,5e-5}; do
    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/finetune.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda:0' --max_iters=5000 --lr_decay_iters=5000 --dtype='float16'
done


# first check best lr - let's first run for shorter time (5000 iters) -> seems like 5e-4!
num_digit=4
num_sample=10000
for lr in {5e-4,1e-4,5e-5}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_ar.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda' --max_iters=2000 --lr_decay_iters=2000 --dtype='float16'
done




num_digit=4
num_sample=5000
for lr in {1e-3,5e-4,1e-4,5e-5}; do # best for simple ar: 5e-4
    python train.py config2/multi_digit/ar_simple/finetune.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --max_iters=5000 --lr_decay_iters=5000 --dtype='float16'
done




####  Let's start finetuning!

num_digit=4
for num_sample in {50000,}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/finetune.py \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
done

num_digit=4
for num_sample in {250,500,1000,5000,10000}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/reverse/finetune.py \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
    --device='cuda' --dtype='float16' --max_iters=10000 --lr_decay_iters=10000
done

for num_sample in {1000,700,500,250,100}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_ar.py \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --max_iters=20000 --lr_decay_iters=20000 --dtype='float16'
done


for num_sample in {250,500,1000,2000,5000}; do
    python train.py config2/multi_digit/ar_simple/finetune.py \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --max_iters=10000 --lr_decay_iters=10000
done


CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_4/plain/ckpt_1000_final.pt" \
--data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name='eval_plain_4digit' \
--dtype='float16'


CUDA_VISIBLE_DEVICES=2 python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_4/reverse/ckpt_250_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name='eval_reverse_4digit' \
--dtype='float16'

CUDA_VISIBLE_DEVICES=0 python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_4/ar/ckpt_100_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name='eval_ar_4digit' \
--dtype='float16'

python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_4/simple/ckpt_500_final.pt" \
--data_type='text' --data_format='algo_reasoning' --operator='+' --algo_reason=True --simple=True --multi_digit=True --multi_model=True \
--num_digit=4 --start="FILE:data/multi_digit/test_4digit_1000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name='eval_ar_simple_4digit'




##########################
# training only on 4-digit
##########################

# first check best lr - 
# for lr in {1e-3,5e-4,1e-4,5e-5}; do

num_digit=4
num_sample=10000
for lr in {1e-5,1e-6}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/finetune_only.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/only_digit_${num_digit}/plain" --train_data_path="only_${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_only_${num_digit}digit_plain_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda:0' --max_iters=10000 --lr_decay_iters=10000 --dtype='float16'
done


# best for reverse: lr = 
# for lr in {1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,1e-7,1e-8}; do
num_digit=4
num_sample=10000
for lr in {1e-6,1e-7,1e-8}; do
    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/finetune_only.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/only_digit_${num_digit}/reverse" --train_data_path="only_${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_only_${num_digit}digit_reverse_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda:0' --max_iters=10000 --lr_decay_iters=10000 --dtype='float16'
done


# for lr in {5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,1e-7,1e-8}; do
# first check best lr 
num_digit=4
num_sample=2000
for lr in {1e-6,1e-7,1e-8}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_only.py \
    --learning_rate=$lr \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/only_digit_${num_digit}/ar" --train_data_path="only_${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_only_${num_digit}digit_ar_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda' --max_iters=20000 --lr_decay_iters=20000 --dtype='float16'
done



# num_digit=4
# num_sample=5000
# for lr in {1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6}; do # best lr: 
#     python train.py config2/multi_digit/ar_simple/finetune_only.py \
#     --learning_rate=$lr \
#     --num_digit=$num_digit --out_dir="out2_multidigit_ft/only_digit_${num_digit}/simple" --train_data_path="only_${num_digit}digit_${num_sample}.txt" \
#     --wandb_run_name="ft_only_${num_digit}digit_simple_${num_sample}_${lr}" --ckpt_path_name="ckpt_${num_sample}_${lr}.pt" \
#     --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
#     --max_iters=10000 --lr_decay_iters=10000 --dtype='float32'
# done





##################################################################
############################# EXP 4. #############################
##################################################################

# simple -> other: aws
# ar -> other: magnetes
# plain / reverse -> other : lee


# Let's first train on 3-digit addition (note that the saved model must have same dtype with our finetune setting) :(
# retraining plain / reverse, since we need a model with context length of 1024! - this is trained on Lee :( 

# Lee's machine
python train.py config2/addition/plain/train_addition_bal_longer_context.py --device='cuda:0'

num_digit=4
for lr in {1e-3,5e-4,1e-4,5e-5}; do
    python train.py config2/multi_digit/plain/finetune_plain_to_reverse.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_plain_to_reverse_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0'

    python train.py config2/multi_digit/plain/finetune_plain_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_plain_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0'

    python train.py config2/multi_digit/plain/finetune_plain_to_ar.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_plain_to_ar_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0'
done

num_digit=4
for lr in {1e-3,}; do
    python train.py config2/multi_digit/plain/finetune_plain_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_plain_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0'
done


python train.py config2/addition/reverse/train_addition_bal_longer_context.py --device='cuda:1'

num_digit=4
for lr in {1e-3,5e-4,1e-4,5e-5}; do
    python train.py config2/multi_digit/reverse/finetune_reverse_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_reverse_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:1'

    python train.py config2/multi_digit/reverse/finetune_reverse_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_reverse_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:1'

    python train.py config2/multi_digit/reverse/finetune_reverse_to_ar.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_reverse_to_ar_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:1'
done



num_digit=4
for lr in {5e-4,}; do
    python train.py config2/multi_digit/reverse/finetune_reverse_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_reverse_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:1'

    python train.py config2/multi_digit/reverse/finetune_reverse_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_reverse_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:1'
done


# Magnetes

num_digit=4
for lr in {1e-4,5e-5}; do
    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/ar/finetune_ar_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'

    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/ar/finetune_ar_to_reverse.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_reverse_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'

    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/ar/finetune_ar_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'
done


num_digit=4
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/ar/finetune_ar_to_plain.py \
--learning_rate=$lr --num_digit=$num_digit \
--wandb_run_name="ft_${num_digit}digit_ar_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
--device='cuda'

num_digit=4
lr=1e-3
CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/ar/finetune_ar_to_simple.py \
--learning_rate=$lr --num_digit=$num_digit \
--wandb_run_name="ft_${num_digit}digit_ar_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
--device='cuda'



num_digit=4
for lr in {1e-3,5e-4}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_ar_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'

    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_ar_to_reverse.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_reverse_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'

    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar/finetune_ar_to_simple.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_ar_to_simple_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda'
done

#  AWS

num_digit=4
for lr in {1e-3,5e-4,1e-4,5e-5}; do
    python train.py config2/multi_digit/ar_simple/finetune_simple_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_simple_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0' --dtype='float32'

    python train.py config2/multi_digit/ar_simple/finetune_simple_to_reverse.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_simple_to_reverse_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0' --dtype='float32'

    python train.py config2/multi_digit/ar_simple/finetune_simple_to_ar.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_simple_to_ar_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0' --dtype='float32'
done


num_digit=4
for lr in {1e-3,}; do
    python train.py config2/multi_digit/ar_simple/finetune_simple_to_plain.py \
    --learning_rate=$lr --num_digit=$num_digit \
    --wandb_run_name="ft_${num_digit}digit_simple_to_plain_${lr}" --ckpt_path_name="ckpt_${lr}.pt" \
    --device='cuda:0' --dtype='float32'
done

# Magnetes
CUDA_VISIBLE_DEVICES=0 python train.py config2/addition/plain/train_addition_bal_longer_context.py --dtype='float16' --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/plain' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='plain_1digit' --start='FILE:data/bal/test_1digit_100.txt'

CUDA_VISIBLE_DEVICES=0 python train.py config2/addition/plain/train_addition_bal_longer_context.py --dtype='float16' --train_data_path='train_2digit_10000_v2.txt' --max_iters=10000 --lr_decay_iters=10000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/plain' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='plain_2digit' --start='FILE:data/bal/test_2digit_10000.txt'


CUDA_VISIBLE_DEVICES=3 python train.py config2/addition/plain/train_addition_bal.py --dtype='float16' --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/plain' --ckpt_path_name='ckpt.pt' --wandb_run_name='plain_1digit' --start='FILE:data/bal/test_1digit_100.txt'

CUDA_VISIBLE_DEVICES=0 python train.py config2/addition/plain/train_addition_bal.py --dtype='float16' --train_data_path='train_2digit_10000_v2.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/plain' --ckpt_path_name='ckpt.pt' --wandb_run_name='plain_2digit' --start='FILE:data/bal/test_2digit_10000.txt'


# Magnetes
CUDA_VISIBLE_DEVICES=2 python train.py config2/addition/reverse/train_addition_bal_longer_context.py --dtype='float16' --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/reverse' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='reverse_1digit' --start='FILE:data/bal/test_1digit_100.txt'

CUDA_VISIBLE_DEVICES=2 python train.py config2/addition/reverse/train_addition_bal_longer_context.py --dtype='float16' --train_data_path='train_2digit_10000_v2.txt' --max_iters=10000 --lr_decay_iters=10000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/reverse' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='reverse_2digit' --start='FILE:data/bal/test_2digit_10000.txt'


CUDA_VISIBLE_DEVICES=2 python train.py config2/addition/reverse/train_addition_bal.py --dtype='float16' --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/reverse' --ckpt_path_name='ckpt.pt' --wandb_run_name='reverse_1digit' --start='FILE:data/bal/test_1digit_100.txt'

CUDA_VISIBLE_DEVICES=1 python train.py config2/addition/reverse/train_addition_bal.py --dtype='float16' --train_data_path='train_2digit_10000_v2.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/reverse' --ckpt_path_name='ckpt.pt' --wandb_run_name='reverse_2digit' --start='FILE:data/bal/test_2digit_10000.txt'


# Magnetes
CUDA_VISIBLE_DEVICES=3 python train.py config2/addition/ar_simple/train_addition.py --dtype='float16' --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/simple' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='simple_1digit' --start='FILE:data/bal/test_1digit_100.txt'

CUDA_VISIBLE_DEVICES=3 python train.py config2/addition/ar_simple/train_addition.py --dtype='float16' --train_data_path='train_2digit_10000.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/simple' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='simple_2digit' --start='FILE:data/bal/test_2digit_10000.txt'

# Lee
python train.py config2/addition/ar/train_addition_ar.py --train_data_path='train_1digit_100.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=1 --multi_digit=True --out_dir='out2_multidigit/digit_1/ar' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='ar_1digit' --start='FILE:data/bal/test_1digit_100.txt'

python train.py config2/addition/ar/train_addition_ar.py --train_data_path='train_2digit_10000.txt' --max_iters=5000 --lr_decay_iters=5000 --num_digit=2 --multi_digit=True --out_dir='out2_multidigit/digit_2/ar' --ckpt_path_name='ckpt_base.pt' --wandb_run_name='ar_2digit' --start='FILE:data/multi_digit/test_2digit_100.txt'





####################################################################
############################# k -> k+1 #############################
####################################################################

# Plain
num_digit=2 # magnetes
for num_sample in {1000,2000,5000,10000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/finetune.py \
    --resume_dir='out2_multidigit/digit_1/plain/ckpt_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

num_digit=3 # magnetes
for num_sample in {1000,2000,5000,10000,20000,50000}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/plain/finetune.py \
    --resume_dir='out2_multidigit/digit_2/plain/ckpt_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda' 
done


num_digit=6 # idunn
for num_sample in {5000,10000,20000,50000,100000,200000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/finetune.py \
    --resume_dir='out2_multidigit/digit_5/plain/ckpt_100000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

num_digit=8 # ida
for num_sample in {5000,10000,20000,50000,100000,200000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/plain/finetune.py \
    --resume_dir='out2_multidigit/digit_7/plain/ckpt_200000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/plain" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_plain_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done



# Reverse 

num_digit=2 # magnetes
for num_sample in {500,1000,2000,3000,4000,5000}; do
    CUDA_VISIBLE_DEVICES=2 python train.py config2/multi_digit/reverse/finetune.py \
    --resume_dir='out2_multidigit/digit_1/reverse/ckpt_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

num_digit=3 # magnetes
for num_sample in {500,1000,2000,3000,4000,5000,7000,10000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/reverse/finetune.py \
    --resume_dir='out2_multidigit/digit_2/reverse/ckpt_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done


num_digit=6 # idunn
for num_sample in {500,1000,2000,5000,7000,10000}; do
    CUDA_VISIBLE_DEVICES=0 python train.py config2/multi_digit/reverse/finetune.py \
    --resume_dir='out2_multidigit/digit_5/reverse/ckpt_7000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done


num_digit=8 # ida
for num_sample in {500,1000,2000,5000,7000,10000,20000,50000}; do
    CUDA_VISIBLE_DEVICES=1 python train.py config2/multi_digit/reverse/finetune.py \
    --resume_dir='out2_multidigit/digit_7/reverse/ckpt_10000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/reverse" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_reverse_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

# Simple
num_digit=2 # magnetes
for num_sample in {500,1000,2000,5000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar_simple/finetune2.py \
    --resume_dir='out2_multidigit/digit_1/simple/ckpt_base_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

num_digit=3 # magnetes
for num_sample in {500,1000,2000,5000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar_simple/finetune2.py \
    --resume_dir='out2_multidigit/digit_2/simple/ckpt_base_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda'
done

num_digit=6 # magnetes
for num_sample in {500,1000,2000,5000,7000,10000,20000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar_simple/finetune2.py \
    --resume_dir='out2_multidigit/digit_5/simple/ckpt_10000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda' --block_size=2048
done

num_digit=8 # magnetes
for num_sample in {500,1000,2000,5000,7000,10000,20000}; do
    CUDA_VISIBLE_DEVICES=3 python train.py config2/multi_digit/ar_simple/finetune2.py \
    --resume_dir='out2_multidigit/digit_7/simple/ckpt_20000_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/simple" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_simple_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_1000.txt" \
    --device='cuda' --block_size=2048
done


# AR
num_digit=2 # lee
for num_sample in {100,200,500,1000}; do
    python train.py config2/multi_digit/ar/finetune2.py \
    --resume_dir='out2_multidigit/digit_1/ar/ckpt_base_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda'
done

num_digit=3 # lee
for num_sample in {100,200,500,1000}; do
    python train.py config2/multi_digit/ar/finetune2.py \
    --resume_dir='out2_multidigit/digit_2/ar/ckpt_base_final.pt' \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda:1'
done

for num_sample in {200,500,1000,2000,3000,4000,5000,7000,10000}; do
num_digit=6 # lee
for num_sample in {4000,}; do
    python train.py config2/multi_digit/ar/finetune2.py \
    --resume_dir='out2_multidigit/digit_5/ar/ckpt_5000_final.pt' --block_size=2048 \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda:1'
done


num_digit=8 # ida
for num_sample in {200,500,1000,2000,5000,7000,10000}; do
    python train.py config2/multi_digit/ar/finetune2.py \
    --resume_dir='out2_multidigit/digit_7/ar/ckpt_5000_final.pt' --block_size=2048 \
    --num_digit=$num_digit --out_dir="out2_multidigit_ft/digit_${num_digit}/ar" \
    --train_data_path="${num_digit}digit_${num_sample}.txt" \
    --wandb_run_name="ft_${num_digit}digit_ar_${num_sample}" --ckpt_path_name="ft_ckpt_${num_sample}.pt" \
    --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
    --device='cuda:1'
done


# For evaluation:
# plain
num_digit=4
python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_${num_digit}/plain/ft_ckpt_5000_final.pt" \
--data_type='text' --data_format='plain' --operator='+' --multi_digit=True --multi_model=True \
--num_digit=$num_digit --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name="eval_plain_${num_digit}digit"

# reverse
num_digit=3
python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_${num_digit}/reverse/ft_ckpt_1000_final.pt" \
--data_type='text' --data_format='reverse' --operator='+' --reverse_c=True --multi_digit=True --multi_model=True \
--num_digit=$num_digit --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name="eval_reverse_${num_digit}digit" --device='cuda'

# simple
num_digit=8
python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_${num_digit}/simple/ft_ckpt_1000.pt" \
--data_type='text' --data_format='algo_reasoning' --algo_reason=True --simple=True --operator='+' --multi_digit=True --multi_model=True \
--num_digit=$num_digit --start="FILE:data/multi_digit/test_${num_digit}digit_10000.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name="eval_simple_${num_digit}digit" --device='cuda:3'

# ar
num_digit=6
python test_addition.py --ckpt_path_name="out2_multidigit_ft/digit_${num_digit}/ar/ft_ckpt_1000_final.pt" \
--data_type='text' --data_format='algo_reasoning' --algo_reason=True --operator='+' --multi_digit=True --multi_model=True \
--num_digit=$num_digit --start="FILE:data/multi_digit/test_${num_digit}digit_100.txt" \
--wandb_log=True --wandb_project='addition_multidigit_ft' --wandb_run_name="eval_ar_${num_digit}digit" --device='cuda:1'

