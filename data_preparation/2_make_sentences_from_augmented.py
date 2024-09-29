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

tokenizer = REMI(params=Path(saved_tokenizer_path))

os.makedirs('../data', exist_ok=True)
sentences_file_path = '../data/midi_sentences.txt'
error_log_file_path = '../data/midi_error_pieces.txt'

# open the txt to write to
with open(sentences_file_path, 'w', encoding='utf-8') as the_file:
    the_file.write('')

# also keep a txt with pieces that are problematic
with open(error_log_file_path, 'w') as the_file:
    the_file.write('')

print('making sentences txt')
midi_files = os.listdir( path_to_dataset )
# test on a set of train files
for i in tqdm(range(len(midi_files))):
    file_path = path_to_dataset + '/' + midi_files[i]
    try:
        t = tokenizer(file_path)
        with open(sentences_file_path, 'a', encoding='utf-8') as the_file:
            the_file.write(' '.join(t[0].tokens) + '\n')
    except:
        print('ERROR with ', midi_files[i])
        with open(error_log_file_path, 'a') as the_file:
            the_file.write(midi_files[i] + '\n')