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

def midi_valid(midi):
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True

midi_paths = list(Path(path_to_dataset).glob("**/*.mid"))
sorted_paths = sorted(midi_paths) # for debugging
print(f"Found {len(midi_paths)} midi files")

tokenizer.tokenize_dataset(        # 2 velocity and 1 duration values
    sorted_paths,
    Path(path_to_tokens),
    midi_valid,
)