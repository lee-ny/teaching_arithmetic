
import argparse
import os
import pickle
import numpy as np
import random
import math

# create a script to "n" generate addition examples so that we have equal number of examples for each number of carry

# example script:
# python create_multidigit_prompt_binary.py --dataset_dir='addition_multidigit_bal_binary_ver1' --create_test=True --total_num_examples=10000 --num_digit=10
# python create_multidigit_prompt_binary.py --dataset_dir='addition_multidigit_bal_binary_ver1' --num_digit=11

class ArgsHelper:
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Creating prompts")

        parser.add_argument("--num_digit", type=int)
        parser.add_argument("--total_num_examples", type=int, default=100)
        parser.add_argument("--reverse", type=bool, default=True)
        parser.add_argument("--dataset_dir", type=str)
        parser.add_argument("--create_test", type=bool, default=False)

        args = parser.parse_args()
        return args

    def get_args(self):
        args = self.parse_arguments()
        # if args.dest_dir is None:
        #     set_dest_dir(args)
        return args


argshelper = ArgsHelper()
args = argshelper.get_args()

num_digits = args.num_digit
total_num_examples = args.total_num_examples
out_dir = args.dataset_dir

def convert_to_binary(num):
    return bin(num).replace("0b", "")

if args.create_test:
    # create test set with random total_num_examples examples
    output_filename = f'{out_dir}/prompt_test_{num_digits}digit_{total_num_examples}.txt'
    if os.path.exists(output_filename):
        print(f'file {output_filename} already exists')
        exit()

    with open(output_filename, 'w') as f:
        num_example = 0
        while num_example < total_num_examples:
            a = random.randint(0, 2**(num_digits) - 1)
            b = random.randint(0, 2**(num_digits) - 1)
            bin_a, bin_b = convert_to_binary(a), convert_to_binary(b)
            f.write(f'${bin_a}+{bin_b}=\n')
            num_example +=1
    
    exit()

# create separate prompt for each digits
for num_digit in range(1, num_digits+1):
    print(num_digit)
    output_filename = f'{out_dir}/prompt_{num_digit}digit_{total_num_examples}.txt'
    if os.path.exists(output_filename):
        print(f'file {output_filename} already exists')
        continue
    
    with open(output_filename, 'w') as f:
        if num_digit < 4:
            for a in range(2 ** num_digit):
                for b in range(2 ** num_digit):
                    if a < 2**(num_digit-1) and b < 2**(num_digit-1):
                        continue
                    bin_a, bin_b = convert_to_binary(a), convert_to_binary(b)
                    f.write(f'${bin_a}+{bin_b}=\n')
        else:
            num_example = 0
            while num_example < total_num_examples:
                a = random.randint(0, 2**(num_digit) - 1)
                b = random.randint(0, 2**(num_digit) - 1)
                # exclude examples with both numbers to have less than num_digit digits
                if a < 2**(num_digit-1) and b < 2**(num_digit-1):
                    continue
                bin_a, bin_b = convert_to_binary(a), convert_to_binary(b)
                f.write(f'${bin_a}+{bin_b}=\n')
                num_example +=1

exit()