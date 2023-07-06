# run files for experimenting the effect of '$'
for num_train in {10000,}; do
    echo $num_train
    python train.py config/train_addition_bal.py \
    --wandb_run_name="add-bal_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out-check-new' \
    --data_type='text' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' \
    --eval_addition_train=True
done



for num_train in {10000,}; do
    echo $num_train
    python train.py config/train_addition_bal.py \
    --wandb_run_name="add-bal-rev_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out-check-new-rev' \
    --data_type='text' --data_format='reverse' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' \
    --eval_addition_train=True
done



# reverse with (reverse) / without $(reverse2)

# for num_train in {500,1000,2000,3000,4000,5000,10000,20000}; do
for num_train in {500,1000,2000,3000}; do
    echo $num_train
    python train.py config2/addition/reverse/train_addition_bal.py \
    --wandb_run_name="add-bal-rev_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_reverse' \
    --data_type='text' --data_format='reverse' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt'
done

for num_train in {4000,5000,10000,20000}; do
    echo $num_train
    python train.py config2/addition/reverse/train_addition_bal.py \
    --wandb_run_name="add-bal-rev_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_reverse' \
    --data_type='text' --data_format='reverse' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' --device='cuda:1'
done

python sample_addition.py \
--wandb_log=True --wandb_project='addition' --wandb_run_name='eval_$-rev-bal_samplenum' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='reverse' --reverse_c=True --dataset='bal' --out_dir='out2/addition_reverse' \
--start='FILE:data/bal/test_10000.txt'


for num_train in {500,1000,2000,3000}; do
    echo $num_train
    python train.py config2/addition/reverse/train_addition_bal.py \
    --wandb_run_name="add-bal-rev2_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_reverse2' \
    --data_type='text' --data_format='reverse2' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt'
done

for num_train in {4000,5000,10000,20000}; do
    echo $num_train
    python train.py config2/addition/reverse/train_addition_bal.py \
    --wandb_run_name="add-bal-rev2_${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_reverse2' \
    --data_type='text' --data_format='reverse2' --reverse_c=True \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' --device='cuda:1'
done


python sample_addition.py \
--wandb_log=True --wandb_project='addition' --wandb_run_name='eval_$-rev2-bal_samplenum' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='reverse2' --reverse_c=True --dataset='bal' --out_dir='out2/addition_reverse2' \
--start='FILE:data/bal/test_10000.txt'



# plain without (plain) / with $(plain2)

# for num_train in {500,1000,2000,3000,4000,5000,10000,20000}; do
for num_train in {500,1000,2000,3000}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_bal.py \
    --wandb_run_name="add-bal-${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_plain' \
    --data_type='text' --data_format='plain' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt'
done

for num_train in {4000,5000,10000,20000}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_bal.py \
    --wandb_run_name="add-bal-${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_plain' \
    --data_type='text' --data_format='plain' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' --device='cuda:1'
done

python sample_addition.py \
--wandb_log=True --wandb_project='addition' --wandb_run_name='eval-bal_samplenum' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='plain' --dataset='bal' --out_dir='out2/addition_plain' \
--start='FILE:data/bal/test_10000.txt'




for num_train in {500,1000,2000,3000}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_bal.py \
    --wandb_run_name="add-bal2-${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_plain2' \
    --data_type='text' --data_format='plain2' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt'
done

for num_train in {4000,5000,10000,20000}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_bal.py \
    --wandb_run_name="add-bal2-${num_train}" \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/addition_plain2' \
    --data_type='text' --data_format='plain2' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt' --device='cuda:1'
done

python sample_addition.py \
--wandb_log=True --wandb_project='addition' --wandb_run_name='eval-bal2_samplenum' \
--plot_sample_acc=True --select='samplenum' \
--data_type='text' --data_format='plain2' --dataset='bal' --out_dir='out2/addition_plain2' \
--start='FILE:data/bal/test_10000.txt'



for num_train in {500,}; do
    echo $num_train
    python train.py config2/addition/plain/train_addition_bal.py \
    --wandb_log=False \
    --ckpt_path_name="ckpt_${num_train}.pt" \
    --out_dir='out2/dummy' \
    --data_type='text' --data_format='plain2' \
    --dataset='bal' --train_data_path="train_3digit_${num_train}.txt" \
    --eval_addition=True --start='FILE:data/bal/test_10000.txt'
done