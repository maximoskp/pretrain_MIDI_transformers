from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import DataCollatorForLanguageModeling
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Load the tokenizer
prefix = '/media/datadisk/datasets/GiantMIDI-PIano/aug'
saved_tokenizer_path = f'{prefix}_REMI_BPE_tokenizer.json'
path_to_dataset = prefix

tokenizer = REMI(params=Path(saved_tokenizer_path))

tokenizer.pad_token = tokenizer.pad_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

files_paths = list(Path(path_to_dataset).glob("**/*.mid"))

# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path("path", "to", "midi_chunks")
split_files_for_training(
    files_paths=files_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=1024,
)

# Create a Dataset, a DataLoader and a collator to train a model
dataset = DatasetMIDI(
    files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)