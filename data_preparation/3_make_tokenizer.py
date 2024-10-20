from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers import normalizers
from transformers import RobertaTokenizerFast

# Initialize the tokenizer with the WordLevel model
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# Use Whitespace pre-tokenizer to split by whitespace
tokenizer.pre_tokenizer = Whitespace()

# Initialize the trainer, without specifying vocab_size
trainer = WordLevelTrainer(
    special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"] # for roberta
)

# Prepare your dataset
sentences_file_path = '../data/midi_sentences.txt'
files = [sentences_file_path]

# Train the tokenizer (vocab_size will be inferred from the data)
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("../data/midi_wordlevel_tokenizer.json")

# save the roberta tokenizer
wrapped_tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer)
wrapped_tokenizer.save_pretrained('../data/midi_wordlevel_tokenizer')