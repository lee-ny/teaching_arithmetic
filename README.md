# Teaching Arithmetic to Small Transformers

## Overview
---
Large language models like GPT-4 exhibit emergent capabilities across general-purpose tasks, such as basic arithmetic, when trained on extensive text data, even though these tasks are not explicitly encoded by the unsupervised, next-token prediction objective. This study investigates how small transformers, trained from random initialization, can efficiently learn arithmetic operations such as addition, multiplication, and elementary functions like square root, using the next-token prediction objective.
We first demonstrate that conventional training data is not the most effective for arithmetic learning, and simple formatting changes can significantly improve accuracy. This leads to sharp phase transitions as a function of training data scale, which, in some cases, can be explained through connections to low-rank matrix completion. Building on prior work, we then train on chain-of-thought style data that includes intermediate step results. Even in the complete absence of pretraining, this approach significantly and simultaneously improves accuracy, sample complexity, and convergence speed.
We also study the interplay between arithmetic and text data during training and examine the effects of few-shot prompting, pretraining, and model scale. Additionally, we discuss length generalization challenges. Our work highlights the importance of high-quality, instructive data that considers the particular characteristics of the next-word prediction objective for rapidly eliciting arithmetic capabilities.

### Notes on the implementation 
---
This codebase is based on the [NanoGPT](https://github.com/karpathy/nanoGPT) repo. We have made some modifications to the codebase to support our experiments.

## Dependencies (tentative)
---
Tested stable dependencies:
* python 3.8.10 (Anaconda)
* PyTorch 2.1.0
* torchvision 0.15.1
* CUDA 11.8
* cuDNN 8.5.0.96
* transformers
* datasets
* tiktoken
* tqdm
* wandb (optional)

## Running Experiments:
---
The main script is `train.py`, to launch the jobs, we provide scripts in `run/`. 
An example of running the code is as follows:
```
python train.py config2/addition/plain/train_addition_bal.py \
--ckpt_path_name="ckpt_10000.pt" \
--out_dir='out/addition_plain' \
--data_type='text' --data_format='plain' \
--dataset='bal' --train_data_path="train_3digit_10000.txt" \
--eval_addition=True --start='FILE:data/bal/test_10000.txt'
```
The argument following `python train.py` is the config file that contains all the hyperparameters and settings for the experiment.
The code trains a model on the addition task using the `plain` formatting technique on the `bal` dataset.
A description of the main arguments is given below.

### Dataset
| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset`      | Dataset to use. (directory name inside `data/`) |
| `data_type` | `binary \| text` .|
| `train_data_path` | Data used for training. If `data_type=binary` this should be a binary file (.bin) else if `data_type=text` this should be a text file (.txt). The data file is `data/{dataset}/{train_data_path}`. |
| `operator` | Operator to be trained on: `+ \| - \| * \| sin \| sqrt`. |
| `data_format` | Formatting techniques to use: `plain \| reverse \| algo_reasoning`. |
| `num_digit` | Number of digits considered. |
| `tokenizer` | Tokenizer used. By default, we use char-level tokenizer `char`. To use the OpenAI tokenizer, set it to `gpt2`.|
| `reverse_c` | Set True to reverse the output (used with `data_format=reverse`). |
| `algo_reason` | Set True to use scratchpad formatting (both detailed/simplified scratchpad). |
| `simple` | Set True to use simplified scratchpad formatting (must be used with `algo_reason=True`). |
| `add_space` | Set True to add a space between each digit.  |
| `vocabulary` | Vocabulary set to consider: `all_ascii_chars` \| `numbers_only` \| `custom_input_data` (vocabulary only consists of characters appearing in the dataset). |

### Model
| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `init_from`   | Select model to train from `scratch`: random init. \| `resume`: resume from `{resume_dir}` (if specified) or `{out_dir}/{ckpt_path_name}` \| `gpt2 \| gpt2-medium \| gpt2-large \| gpt2-xl`: pretrained GPT-2 models. |
| `n_layer`     | Number of self-attention layers. |
| `n_head`      | Number of heads. |
| `n_embd`      | Dimension for embedding. |
| `block_size`  | Context length. |
| `dropout`     | Dropout rate. |


### Learning Rate Policy
| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `learning_rate` | Max learning rate (after warmup) that will be used for the training process. |
| `batch_size` | Batch size for the optimizer (AdamW). |
| `gradient_accumulation_steps` | Used to simulate larger batch sizes. |
| `max_iters` | Total number of training iterations. |
| `warmup_iters` | Number of iterations to warm up for (learning rate will increase linearly to `learning_rate` over `warmup_iters` iterations). |
| `lr_decay_iters` | Number of iterations to decay the learning rate (using cosine learning rate decay). |
| `min_lr` | Minimum learning rate. Automatically set to `learning_rate/10` if not specified. |
| `weight_decay` | Weight decay coefficient. |
| `beta1`, `beta2` | Coefficients used for computing running averages of gradient and its square. |

### Evaluation and Checkpointing
| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `eval_addition` | Set True to evaluate the performance on the arithmetic task (given by `operator`) that is being trained on over the test data `start`. |
| `start` | Test data to be evaluated. Prepend with `"FILE:"` to specify a specific test data (either .txt or .bin file depending on `data_type`) to be evaluated. Else, `start` is regarded as a test sequence to directly be input to the model. |
| `multi_digit` | Set True to evaluate test accuracy on each digit (1 to `num_digit`) test data |
| `eval_addition_train` | Set True to evaluate the train data used for training. |
| `eval_text`   | Set True to evaluate the perplexity on the text `eval_text_data`. |
| `eval_addition_ar`   | Set True to evaluate the performance on scratchpad methods over the test data `start_ar`. |
| `eval_other`   | Set True to evaluate the performance on the arithmetic task (given by `other_operator`) over the test data (`start_other`). This is used to evaluate the performance that is not identical to the `operator`, which the model is being trained on.  |
| `out_dir`   | Directory to save the model. |
| `ckpt_path_name`     | Filename of the saved model (inside `{out_dir}/`). |
| `eval_interval`     | Number of iteration intervals at which evaluations will be performed. |
| `eval_iters`     | Number of batches used to estimate the loss |
| `log_interval`     | Number of iteration intervals at which the loss is printed. |
| `always_save_checkpoint`     | Set True to always save a checkpoint after each eval. |

### Configs
Note that the workflow is managed by specifying the above arguments using the config files specified in the `config/, config2/, config_gpt2` directory and running them with modifications as provided in the scripts in `run/`, `run_gpt2/`.

### Sample Config
```
# ===== Evaluation and Checkpointing ===== #
out_dir = 'out2/addition_plain'
eval_interval = 250 
eval_iters = 200
log_interval = 10 
always_save_checkpoint = False

# ===== Wandb logging ===== #
wandb_log = True # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'addition_plain'

# ===== Dataset ===== #
data_type='text'
data_format='plain'
operator='+'
dataset = 'bal'
batch_size = 256
train_data_path = 'train_3digit_10000.txt'
ckpt_path_name = 'ckpt_10000.pt'
eval_addition = True
start = "FILE:data/bal/test_10000.txt"
eval_addition_train = True

# ===== NanoGPT model configuration ===== #
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
block_size = 256 # context of up to 256 previous characters

# ===== Learning Rate Policy ===== #
learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
beta2 = 0.99
warmup_iters = 100

# ===== Device ===== #
device='cuda:0'
```
