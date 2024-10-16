from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import DataCollatorForLanguageModeling
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

# Load the tokenizer
prefix = '/media/datadisk/datasets/GiantMIDI-PIano/aug'
saved_tokenizer_path = f'{prefix}_REMI_BPE_tokenizer.json'
path_to_dataset = prefix
path_to_train_splits = f'{prefix}_splits_REMI_BPE/train/midis'
path_to_valid_splits = f'{prefix}_splits_REMI_BPE/valid/midis'

tokenizer = REMI(params=Path(saved_tokenizer_path))

tokenizer.pad_token = tokenizer.pad_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

files_paths = list(Path(path_to_dataset).glob("**/*.mid"))
train_paths = list(Path(path_to_train_splits).glob("**/*.mid"))
valid_paths = list(Path(path_to_valid_splits).glob("**/*.mid"))

# Create a Dataset, a DataLoader and a collator to train a model
dataset = DatasetMIDI(
    files_paths=train_paths,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

training_args = TrainingArguments(
    output_dir='/models/robertaGM',
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()