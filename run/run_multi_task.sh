python train_multi_task.py config2/multi_task/train_multi_task.py --device='cuda:1'

python train_multi_task.py config2/multi_task/train_multi_task2.py --device='cuda:1'

python train_multi_task.py config2/multi_task/train_multi_task_ar.py --device='cuda:1'

# evaluate zero-shot # The train curves already test on the full test dataset so we simply take the final test accuracy from the train result file

# evaluate few-shot
python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_add_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/add/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_add_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/add/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_add_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/add/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_add_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/add/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_add_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/add/few_shot_prefix/3shot_1.txt'


python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sub_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sub/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sub_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sub/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sub_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/sub/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sub_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/sub/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sub_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/sub/few_shot_prefix/3shot_1.txt'


python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_mul_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/mul/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_mul_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/mul/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_mul_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/mul/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_mul_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/mul/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_mul_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/mul/few_shot_prefix/3shot_1.txt'


python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sin_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sin/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sin_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sin/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sin_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/sin/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sin_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/sin/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sin_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/sin/few_shot_prefix/3shot_1.txt'


python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sqrt_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sqrt/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sqrt_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/sqrt/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sqrt_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/sqrt/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sqrt_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/sqrt/few_shot_prefix/3shot_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='fewshot_sqrt_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/sqrt/few_shot_prefix/3shot_1.txt'


# evaluate text-prompt :
python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='text_add' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='+' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/word_prefix/phrase1_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='text_sub' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='-' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_3digit_10000.txt" --prompt_dir='prompts/word_prefix/phrase1_1.txt'

python sample_addition_fewshot2.py --device='cuda:0' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='text_mul' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='*' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/bal/test_multiplication_7000.txt" --prompt_dir='prompts/word_prefix/phrase1_1.txt'

python sample_addition_fewshot2.py --device='cuda:1' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='text_sin' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sin' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sin/test_sin_10000.txt" --prompt_dir='prompts/word_prefix/phrase1_1.txt'

python sample_addition_fewshot2.py --device='cuda:1' \
--wandb_log=True --wandb_project='mixed_task' --wandb_run_name='text_sqrt' --out_dir='out2/mixed_task/mixed_task_256' \
--algo_reason=False --dataset='bal' --data_type='text' --data_format='plain' --operator='sqrt' \
--plot_sample_acc=False --select='fewshot' --fewshot=True --multiple_set_per_prompt=True \
--start="FILE:data/sqrt/test_sqrt_10000.txt" --prompt_dir='prompts/word_prefix/phrase1_1.txt'


# evaluate noisy-prompt # TODO: not sure if we need it rn