from miditok import REMI, TokenizerConfig, TokTrainingIterator
from pathlib import Path

# Tokenizer configuration
PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NUM_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = True # changed
USE_RESTS = False
USE_TEMPOS = True # changed
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NUM_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "num_velocities": NUM_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "num_tempos": NUM_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

### full dataset
prefix = '/media/datadisk/datasets/GiantMIDI-PIano/aug'
path_to_dataset = prefix
path_to_tokens = f'{prefix}_REMI_noBPE/'
path_to_tokens_bpe = f'{prefix}_REMI_BPE/'
path_to_tokenizer_config = f'{prefix}_REMI_BPE_tokenizer.json'

midi_paths = Path(path_to_dataset)

train_iterator = TokTrainingIterator(
    tokenizer=tokenizer,
    files_paths=list(Path(path_to_dataset).glob('**/*.mid'))
)

tokenizer.train(
    vocab_size=10000,
    model='BPE',
    iterator=train_iterator
)

# Save tokenizer
tokenizer.save_params(Path(path_to_tokenizer_config))