# training
### Task1
## Toy
# python train.py config/ar_simple/train_addition.py --wandb_log=False --n_layer=3 --n_head=3

# python train.py config/ar_simple/train_addition.py --wandb_run_name='simple' --ckpt_path_name='ckpt_lr_1e-3.pt'

# python train.py config/ar_simple/train_addition_randomA.py --wandb_run_name='simple_randomA' --ckpt_path_name='ckpt_lr_1e-3.pt'

# python train.py config/ar_simple/train_addition_randomC.py --wandb_run_name='simple_randomC' --ckpt_path_name='ckpt_lr_1e-3.pt'

python train.py config/ar_simple/train_addition_randomboth.py --wandb_run_name='simple_randomBoth' --ckpt_path_name='ckpt_lr_1e-3.pt'

python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_ar_simple' \
--out_dir='out/ar_simple' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
--algo_reason=True --dataset='ar_simple' \
--prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.01.txt" \
--prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
--prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"

python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_ar_simple_randomA' \
--out_dir='out/ar_simple_randomA' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
--algo_reason=True --dataset='ar_simple_randomA' \
--prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.01.txt" \
--prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
--prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"


python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_ar_simple_randomC' \
--out_dir='out/ar_simple_randomC' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
--algo_reason=True --dataset='ar_simple_randomC' \
--prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.01.txt" \
--prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
--prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"


python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_ar_simple_randomBoth' \
--out_dir='out/ar_simple_randomBoth' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
--algo_reason=True --dataset='ar_simple_randomboth' \
--prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.01.txt" \
--prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
--prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"



python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_ar_simple' \
--out_dir='out/ar_simple' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
--algo_reason=True --dataset='ar_simple' \
--prompt_overall="FILE:data/algo_reasoning/prompt_train.txt" \



# python sample_addition.py --out_dir='out/algo_reasoning-addition' --ckpt_path_name=ckpt_lr1e-4_final.pt \
# --algo_reason=True --dataset='algo_reasoning' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 1654+9873"


# python sample.py --out_dir='out/algo_reasoning-addition' --ckpt_path_name=ckpt_lr1e-4_final.pt \
# --dataset='algo_reasoning' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# [8,4,6,5] + [3,5,4,1] , A=[] , C=0 , 5+1+0=6 , A->6 , C->0
# [8,4,6] + [3,5,4] , A=[6] , C=0 , 6+4+0=10 , A->0 , C->1
# [8,4] + [3,5] , A=[0,6] , C=1 , 4+5+1=10 , A->0 , C->1
# [8] + [3] , A=[0,0,6] , C=1 , 8+3+1=12 , A->2 , C->1
# [] + [] , A=[2,0,0,6] C=1 , END
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# [] + [] , A=[5,5,4,4] C=0 , END
# </scratch>
# 5 5 4 4
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# [] + [] , A=[5,5,4,4] C=0 , END
# </scratch>
# 5 5 4 4
# Input:
# 1654+9873
# "

##########################################################################
################## algorithmic reasoning with lines ######################
##########################################################################

# python train.py config/algo_reasoning/train_addition_lines.py --wandb_run_name='lines_lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'

# python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_algo_reason_lines' --device='cuda:1' \
# --out_dir='out/algo_reasoning-addition-lines' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --algo_reason=True --dataset='algo_reasoning_lines' \
# --prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"


# python sample.py --out_dir='out/algo_reasoning-addition-lines' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_lines' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# total 5 lines.
# line1 , [8,4,6,5] + [3,5,4,1] , A=[] , C=0 , 5+1+0=6 , A->6 , C->0
# line2 , [8,4,6] + [3,5,4] , A=[6] , C=0 , 6+4+0=10 , A->0 , C->1
# line3 , [8,4] + [3,5] , A=[0,6] , C=1 , 4+5+1=10 , A->0 , C->1
# line4 , [8] + [3] , A=[0,0,6] , C=1 , 8+3+1=12 , A->2 , C->1
# line5 , [] + [] , A=[2,0,0,6] C=1 , END
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# total 5 lines.
# line1 , [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# line2 , [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# line3 , [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# line4 , [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# line5 , [] + [] , A=[5,5,4,4] C=0 , END
# </scratch>
# 5 5 4 4
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# total 5 lines.
"


# python sample.py --out_dir='out/algo_reasoning-addition-lines' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_lines' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 19+91
# Target:
# "

# python sample.py --out_dir='out/algo_reasoning-addition-lines' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_lines' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has "


##########################################################################
########### algorithmic reasoning with  flipped line numbers #############
##########################################################################

# python train.py config/algo_reasoning/train_addition_lines_flipped.py --wandb_run_name='lines_flipped_lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt'

# python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_algo_reason_lines_flipped' --device='cuda:0' \
# --out_dir='out/algo_reasoning-addition-flipped' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --algo_reason=True --dataset='algo_reasoning_lines_flipped' \
# --prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"

# python sample.py --out_dir='out/algo_reasoning-addition-flipped' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_lines' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# total 5 lines.
# line5 , [8,4,6,5] + [3,5,4,1] , A=[] , C=0 , 5+1+0=6 , A->6 , C->0
# line4 , [8,4,6] + [3,5,4] , A=[6] , C=0 , 6+4+0=10 , A->0 , C->1
# line3 , [8,4] + [3,5] , A=[0,6] , C=1 , 4+5+1=10 , A->0 , C->1
# line2 , [8] + [3] , A=[0,0,6] , C=1 , 8+3+1=12 , A->2 , C->1
# line1 , [] + [] , A=[2,0,0,6] C=1 , END
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# total 5 lines.
# line5 , [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# line4 , [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# line3 , [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# line2 , [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# line1 , [] + [] , A=[5,5,4,4] C=0 , END
# </scratch>
# 5 5 4 4
# "


##########################################################################
############# algorithmic reasoning with flipped contents ################
##########################################################################

# python train.py config/algo_reasoning/train_addition_lines_flipped_contents.py --wandb_run_name='lines_flipped_contents_lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt' --device='cuda:1'

# python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_algo_reason_lines_flipped_contents' --device='cuda:0' \
# --out_dir='out/algo_reasoning-addition-flipped-contents' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --algo_reason=True --dataset='algo_reasoning_lines-flipped_contents' \
# --prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"

# python sample.py --out_dir='out/algo_reasoning-addition-flipped-contents' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_lines_flipped_contents' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# total 5 lines.
# line5 , [] + [] , A=[2,0,0,6] C=1 , END
# line4 , [8] + [3] , A=[0,0,6] , C=1 , 8+3+1=12 , A->2 , C->1
# line3 , [8,4] + [3,5] , A=[0,6] , C=1 , 4+5+1=10 , A->0 , C->1
# line2 , [8,4,6] + [3,5,4] , A=[6] , C=0 , 6+4+0=10 , A->0 , C->1
# line1 , [8,4,6,5] + [3,5,4,1] , A=[] , C=0 , 5+1+0=6 , A->6 , C->0
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# total 5 lines.
# line5 , [] + [] , A=[5,5,4,4] C=0 , END
# line4 , [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# line3 , [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# line2 , [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# line1 , [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# </scratch>
# 5 5 4 4

##########################################################################
############# algorithmic reasoning with random contents ################
##########################################################################

# python train.py config/algo_reasoning/train_addition_random.py --wandb_run_name='random_lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt' --device='cuda:0'

# python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_algo_reason_random' --device='cuda:1' \
# --out_dir='out/algo_reasoning-addition-random' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --algo_reason=True --dataset='algo_reasoning_random' \
# --prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"


# python sample.py --out_dir='out/algo_reasoning-addition-random' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_random' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# [2] + [4] , A=[9] , C=0 , 5+6+0=15 , A->6 , C->0
# [5,4] + [6,9,4] , A=[6] , C=0 , 6+4+1=4 , A->0 , C->1
# [4,5] + [3] , A=[5,7] , C=1 , 4+8+0=10 , A->0 , C->1
# [4,7] + [3] , A=[7,5] , C=1 , 7+7+0=14 , A->2 , C->1
# [7,1] + [7] , A=[7] C=1 , END
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# [3,0] + [1] , A=[3] , C=0 , 3+3+1=11 , A->4 , C->1
# [8] + [0,9,8] , A=[1,6] , C=1 , 0+0+1=5 , A->4 , C->1
# [4] + [5] , A=[6,5] , C=1 , 9+2+1=14 , A->5 , C->1
# [7,5] + [9,3] , A=[1] , C=1 , 4+0+0=10 , A->5 , C->0
# [0,9,5] + [8,5,7] , A=[9,6,9] C=0 , END
# </scratch>
# 5 5 4 4



##########################################################################
############# algorithmic reasoning with trained counting ################
##########################################################################

# python train.py config/algo_reasoning/train_addition_counting.py --wandb_run_name='counting_lr_1e-3' --ckpt_path_name='ckpt_lr_1e-3.pt' --device='cuda:0'

# python evaluate_additions.py --wandb_project='algo_reasoning' --wandb_run_name='eval_algo_reason_counting' --device='cuda:1' \
# --out_dir='out/algo_reasoning-addition-counting' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --algo_reason=True --dataset='algo_reasoning_counting' \
# --prompt_overall="FILE:data/algo_reasoning/prompt_addition_test_0.001.txt" \
# --prompt1="FILE:data/algo_reasoning/prompt_addition_test_1.txt" \
# --prompt2="FILE:data/algo_reasoning/prompt_addition_test_2_0.1.txt"

# python sample.py --out_dir='out/algo_reasoning-addition-counting' --ckpt_path_name=ckpt_lr_1e-3_final.pt \
# --dataset='algo_reasoning_counting' --wandb_log=False --num_samples=1 --max_new_tokens=500 \
# --start="Input:
# 8465+3541
# Target:
# <scratch>
# [8,4,6,5] has 4 digits.
# [3,5,4,1] has 4 digits.
# [8,4,6,5] + [3,5,4,1] , A=[] , C=0 , 5+1+0=6 , A->6 , C->0
# [8,4,6] + [3,5,4] , A=[6] , C=0 , 6+4+0=10 , A->0 , C->1
# [8,4] + [3,5] , A=[0,6] , C=1 , 4+5+1=10 , A->0 , C->1
# [8] + [3] , A=[0,0,6] , C=1 , 8+3+1=12 , A->2 , C->1
# [] + [] , A=[2,0,0,6] C=1 , END
# </scratch>
# 1 2 0 0 6
# Input:
# 1946+3598
# Target:
# <scratch>
# [1,9,4,6] has 4 digits.
# [3,5,9,8] has 4 digits.
# [1,9,4,6] + [3,5,9,8] , A=[] , C=0 , 6+8+0=14 , A->4 , C->1
# [1,9,4] + [3,5,9] , A=[4] , C=1 , 4+9+1=14 , A->4 , C->1
# [1,9] + [3,5] , A=[4,4] , C=1 , 9+5+1=15 , A->5 , C->1
# [1] + [3] , A=[5,4,4] , C=1 , 1+3+1=5 , A->5 , C->0
# [] + [] , A=[5,5,4,4] C=0 , END
# </scratch>
# 5 5 4 4