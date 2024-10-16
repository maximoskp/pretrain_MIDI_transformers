from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import DataCollatorForLanguageModeling
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path
from random import shuffle
import os
from tqdm import tqdm

# Load the tokenizer
prefix = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2/aug/midis'
saved_tokenizer_path = f'{prefix}_REMI_BPE_tokenizer.json'
path_to_dataset = prefix
path_to_train_splits = f'{prefix}_splits_REMI_BPE/train/midis'
path_to_valid_splits = f'{prefix}_splits_REMI_BPE/valid/midis'

tokenizer = REMI(params=Path(saved_tokenizer_path))

# open the txt to write to
with open('train_sentences.txt', 'w') as the_file:
    the_file.write('')

# also keep a txt with pieces that are problematic
with open('train_error_pieces.txt', 'w') as the_file:
    the_file.write('')

print('making train sentences txt')
train_files = os.listdir( path_to_train_splits )
# test on a set of train files
for i in tqdm(range(len(train_files))):
    file_path = path_to_train_splits + '/' + train_files[i]
    try:
        t = tokenizer(file_path)
        with open('train_sentences.txt', 'a') as the_file:
            the_file.write(' '.join(t[0].tokens) + '\n')
    except:
        print('ERROR with ', train_files[i])
        with open('train_error_pieces.txt', 'a') as the_file:
            the_file.write(train_files[i] + '\n')

# open the txt to write to
with open('valid_sentences.txt', 'w') as the_file:
    the_file.write('')

# also keep a txt with pieces that are problematic
with open('valid_error_pieces.txt', 'w') as the_file:
    the_file.write('')

print('making valid sentences txt')
valid_files = os.listdir( path_to_valid_splits )
# test on a set of valid files
for i in tqdm(range(len(valid_files))):
    file_path = path_to_valid_splits + '/' + valid_files[i]
    try:
        t = tokenizer(file_path)
        with open('valid_sentences.txt', 'a') as the_file:
            the_file.write(' '.join(t[0].tokens) + '\n')
    except:
        print('ERROR with ', valid_files[i])
        with open('valid_error_pieces.txt', 'a') as the_file:
            the_file.write(valid_files[i] + '\n')