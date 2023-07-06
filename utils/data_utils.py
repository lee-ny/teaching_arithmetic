import random
import os

def make_addition_examples(pad=True):
    input_file_path = os.path.join(os.path.dirname(__file__), 'addition/add_examples.txt')
    # if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        for i in range(10000):
            a, b = random.randint(0,999), random.randint(0,999)
            c = a + b
            if pad:
                f.write(f"{a:03}+{b:03}={c}\n")
            else:
                f.write(f"{a}+{b}={c}\n")


def make_addition_test_examples(pad=True, p=1):
    if pad:
        input_file_path =  os.path.join('prompt',f'prompt_addition_pad_test_{p}.txt') # os.path.join(os.path.dirname(__file__), 'prompt_addition_pad_test.txt')
    else:
        input_file_path = os.path.join('prompt', f'prompt_addition_test_{p}.txt')
    print(input_file_path)
    # if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        for a in range(999):
            for b in range(999):
                pp = random.uniform(0, 1)
                if pp > p:
                    continue
                c = a + b
                if pad:
                    f.write(f"{a:03}+{b:03}=\n")
                else:
                    f.write(f"{a}+{b}=\n")

def make_addition_test_subexamples(pad=True, p=1):
    if pad:
        test='prompt/prompt_addition_pad_nonoverlap.txt'
        input_file_path =  os.path.join('prompt',f'prompt_addition_pad_test_{p}.txt') # os.path.join(os.path.dirname(__file__), 'prompt_addition_pad_test.txt')
    else:
        test='prompt/prompt_addition_nonoverlap.txt'
        input_file_path = os.path.join('prompt', f'prompt_addition_test_{p}.txt')
    print(input_file_path)
    # if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        with open(test, 'r') as ft:
            for line in ft.readlines():
                pp = random.uniform(0, 1)
                if pp > p:
                    continue
                else:
                    f.write(line)


def mix_addition_examples(): # TODO:
    input_file_path = os.path.join(os.path.dirname(__file__), 'add_examples.txt')
    # if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        for i in range(10000):
            a, b = random.randint(0,1000), random.randint(0,1000)
            c = a + b
            f.write(f"{a}+{b}={c}\n")

    # words = []
    # for filename in ['File1', 'File2']:
    # with open(filename, 'r') as file: 
    #     # Opening the file using the with statement will ensure that it is properly
    #     # closed when your done.

    #     words.append((line.strip() for line in file.readlines()))
    #     # The readlines method returns a list of the lines in the file

    #     random.shuffle(words[-1])
    #     # Shuffle will randomize them
    #     # The -1 index refers to the last item (the one we just added)