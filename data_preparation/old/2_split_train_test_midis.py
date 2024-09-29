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
path_to_train_splits = f'{prefix}_splits_REMI_BPE/train/midis'
path_to_valid_splits = f'{prefix}_splits_REMI_BPE/valid/midis'

max_seq_len = 1024

tokenizer = REMI(params=Path(saved_tokenizer_path))

files_paths = list(Path(path_to_dataset).glob("**/*.mid"))
shuffle(files_paths)

total_num_files = len(files_paths)
num_files_valid = round(total_num_files * 0.10)

midi_paths_valid = files_paths[:num_files_valid]
midi_paths_train = files_paths[num_files_valid:]

# Split MIDIs into smaller chunks for validation
dataset_chunks_dir = Path(path_to_valid_splits)
split_files_for_training(
    files_paths=midi_paths_valid,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=max_seq_len,
)


# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path(path_to_train_splits)
split_files_for_training(
    files_paths=midi_paths_train,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=max_seq_len,
)