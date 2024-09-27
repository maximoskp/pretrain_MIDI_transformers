from miditok.data_augmentation import augment_dataset
import os
from pathlib import Path

# prefix = '/media/datadisk/datasets/GiantMIDI-PIano'
prefix = '/media/maindisk/maximos/data/GiantMIDI-PIano/midis_v1.2'

path_to_dataset = prefix
midi_paths = Path(path_to_dataset)

path_to_aug = prefix + '/aug'
os.makedirs(path_to_aug, exist_ok=True)

augment_dataset(
    data_path=midi_paths,
    out_path=path_to_aug,
    pitch_offsets=list(range(-6,5,1)),
    velocity_offsets=[-4, 4],
    duration_offsets=[-0.5, 0.5],
)