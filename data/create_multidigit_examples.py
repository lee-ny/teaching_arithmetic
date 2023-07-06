
import argparse
import os
import pickle
import numpy as np
import random
import math

# create a script to "n" generate addition examples so that we have equal number of examples for each number of carry
# python create_multidigit_examples.py --total_num_examples=10000 --num_digit=7 --ver=1

class ArgsHelper:
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Creating train dataset")

        parser.add_argument("--num_digit", type=int)
        parser.add_argument("--total_num_examples", type=int)
        parser.add_argument("--ver", type=int, default="1")

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
ver = args.ver
total_num_examples = args.total_num_examples
target_num_carry_examples = math.ceil(total_num_examples / (num_digits+1)) # number of examples for each number of carry

# this list will contain number of examples for each number carry
# i.e., num_carry_list[0] = 1000 means that there are 1000 examples with 0 carry
# or num_carry_list[i] = 1000 means that there are 1000 examples with i carry
num_carry_list = [0 for i in range(num_digits+1)]

def numCarryOps(a, b):
    a,b=int(a),int(b)
    def digitSum(n):
        return sum(map(int,str(n)))
    # assert(a >= 0); assert(b >= 0);
    return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)

def reverse_string(a: str) -> str:
    return a[::-1]

if ver == 1:
    # Ver 1.
    # Let's make all p_i's equal to total_num_examples // num_digits
    num_per_digit = total_num_examples // num_digits
    if num_per_digit > 100:
        num_per_digit = (total_num_examples-100) // (num_digits-1)
        p_list = [100] + [num_per_digit for i in range(2, num_digits+1)]
        p_list[1] += total_num_examples - sum(p_list) # give the remaining examples to p_2
    else:
        p_list = [num_per_digit for i in range(1, num_digits+1)]
        p_list[0] += total_num_examples - sum(p_list) # give the remaining examples to p_2
        
elif ver == 2:
    # Ver 2. 
    # Let's make p_i increasing by x10, and p_1 = 1 so that we include all 1-digit exmples
    # That'd be p_1 = 100, p_2 = p, p_3 = 10p, p_4 = 100p, ... and sum(p_i) = total_num_examples
    # Then, 100 + p*(10^(num_digits-1)-1)/9 = total_num_examples -> p = (total_num_examples - 100) * 9 / (10^(num_digits-1)-1)
    p = (total_num_examples - 100) * 9 / (10**(num_digits-1)-1)
    p_list = [100] + [int(10 ** (i-2)*p) for i in range(2, num_digits+1)]
    p_list[1] += total_num_examples - sum(p_list) # give the remaining examples to p_2

print('number of examples for each digit: ', p_list)
p_list_percentage = [100* p_list[i]/(10**(2*(i+1))) for i in range(len(p_list))]

# if folder 'addition_multidigit' does not exist, create it
out_dir = f'addition_multidigit_bal_ver{ver}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

output_filename = f'{out_dir}/add_examples_{num_digits}_{total_num_examples}.txt'
# if the file already exists, don't create it again
if os.path.exists(output_filename):
    print(f'file {output_filename} already exists')

else:
    # append info to the file info_file = f'{out_dir}/info.txt'
    info_file = f'{out_dir}/info.txt'
    f_info = open(info_file, 'a')
    f_info.write(f'creating {num_digits} examples: \nnum_examples for each carry: {target_num_carry_examples}\nnum_examples for each digit: {p_list}\n')
    f_info.write(f'percentage of exmamples for each digit: {p_list_percentage}\n')

    with open(output_filename, 'w') as f:
        # have a big while loop
        # sample in increasing order of number of digits
        # create a target_num_examples for each number of digits

        # but this won't ensure balanced dataset
        # so choose p_i's to be the number of examples of [0, 10^(i)-1] that we select
        # so that the total number of examples is total_num_examples

        num_example = 0
        for num_digit in range(1, num_digits+1):
            num_digit_example = 0
            print(num_digit, p_list[num_digit-1], num_example, num_carry_list)
            f_info.write(f'{num_digit},{p_list[num_digit-1]}, {num_example}, {num_carry_list}\n')
            while num_digit_example < p_list[num_digit-1] and num_example < total_num_examples:
                # generate a random number between 0 and 10^(i+1) - 1
                a = random.randint(0, 10**(num_digit) - 1)
                b = random.randint(0, 10**(num_digit) - 1)
                c = a + b

                # count number of carries in c
                num_carry = numCarryOps(a, b)
                if num_carry_list[num_carry] < target_num_carry_examples:
                    c_rev = reverse_string(str(c))
                    # write the example to file
                    f.write(f'${a}+{b}={c_rev}$\n')
                    # increment num_carry_list[num_carry]
                    num_carry_list[num_carry] += 1
                    num_digit_example += 1
                    num_example += 1
                else:
                    continue

    print(f'created {num_example} number of examples and saved to {output_filename}')
    f_info.write(f'created {num_example} number of examples and saved to {output_filename}\n')

    print(num_carry_list)

    # shuffle lines of the file and save to a new file
    output_filename_shuffled = f'{out_dir}/add_examples_{num_digits}_{total_num_examples}_shuffled.txt'
    with open(output_filename, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        with open(output_filename_shuffled, 'w') as f:
            f.writelines(lines)

output_filename_shuffled = f'{out_dir}/add_examples_{num_digits}_{total_num_examples}_shuffled.txt'
# create .bin file and meta file
with open(output_filename_shuffled, 'r') as f:
    data = f.read()        

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data) # 130,023
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(f'{out_dir}/train_{num_digits}_{total_num_examples}.bin')
val_ids.tofile(f'{out_dir}/val_{num_digits}_{total_num_examples}.bin')

# save the meta information as well, to help us encode/decode later
if not os.path.exists(f'{out_dir}/meta.pkl'):
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(f'{out_dir}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
