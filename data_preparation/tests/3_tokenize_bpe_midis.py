from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import DataCollatorForLanguageModeling
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path
from random import shuffle

# Load the tokenizer
prefix = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2/aug/midis'
saved_tokenizer_path = f'{prefix}_REMI_BPE_tokenizer.json'
path_to_dataset = prefix

path_to_valid_splits = f'{prefix}_splits_REMI_BPE/valid/midis'
path_to_tokens = f'{prefix}_REMI_noBPE/'
path_to_tokens_bpe = f'{prefix}_REMI_BPE/'
path_to_tokenizer_config = f'{prefix}_REMI_BPE_tokenizer.json'

max_seq_len = 1024

tokenizer = REMI(params=Path(saved_tokenizer_path))

tokenizer.pad_token = tokenizer.special_tokens[0]
tokenizer.mask_token = tokenizer.special_tokens[1]

# Load pretrained tokenizer
path_to_bpe = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2/aug/midis_REMI_BPE_tokenizer.json'
bpe_tokenizer = REMI(params=path_to_bpe)
# import pdb; pdb.set_trace()
# Applies BPE to the previous tokens
bpe_tokenizer.train(Path(path_to_tokens), Path(path_to_tokens_bpe))