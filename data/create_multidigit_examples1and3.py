import os
import pickle
import numpy as np
import random


def numCarryOps(a, b):
    a,b=int(a),int(b)
    def digitSum(n):
        return sum(map(int,str(n)))
    # assert(a >= 0); assert(b >= 0);
    return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)

def reverse_string(a: str) -> str:
    return a[::-1]

num_carry_list = [0 for i in range(4)]

input_filename = 'addition_dollar_reverse_curr_bal2/add_examples.txt'

# if folder 'addition_multidigit' does not exist, create it
out_dir = f'addition_multidigit_bal_ver1'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

output_filename = f'{out_dir}/add_examples_1and3.txt'
# if the file already exists, don't create it again
if os.path.exists(output_filename):
    print(f'file {output_filename} already exists')
else:
    # append info to the file info_file = f'{out_dir}/info.txt'
    with open(input_filename, 'r') as f:
        lines = f.readlines()
    with open(output_filename, 'w') as fw:
        for line in lines:
            line = line.replace('$', '')
            line = line.replace('\n', '')
            a, b = line.split('+')
            b, c = b.split('=')

            if ( 10 <= int(a) and int(a) < 100 ) or ( 10 <= int(b) and int(b) < 100 ):
                continue
            fw.write(f'${a}+{b}={c}$\n')
            num_carry = numCarryOps(a, b)
            num_carry_list[num_carry] += 1

    num_example = sum(num_carry_list)
    print(f'created {num_example} number of examples and saved to {output_filename}')
    print(num_carry_list)

    # shuffle lines of the file and save to a new file
    output_filename_shuffled = f'{out_dir}/add_examples_1and3_shuffled.txt'
    with open(output_filename, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        with open(output_filename_shuffled, 'w') as f:
            f.writelines(lines)

output_filename_shuffled = f'{out_dir}/add_examples_1and3_shuffled.txt'
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
train_ids.tofile(f'{out_dir}/train_1and3.bin')
val_ids.tofile(f'{out_dir}/val_1and3.bin')

# save the meta information as well, to help us encode/decode later
if not os.path.exists(f'{out_dir}/meta.pkl'):
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(f'{out_dir}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
