# datasets/dreamer.py
from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DREAMER_CHANNEL_LOCATION_DICT

def load_dreamer_dataset(mat_path="/workspace/DREAMER.mat", io_path=".torcheeg/dreamer_cache"):
    return DREAMERDataset(mat_path=mat_path,
                        io_path=io_path,
                         online_transform=transforms.Compose([
                             transforms.To2d(),
                             transforms.ToTensor()
                         ]),
                         label_transform=transforms.Compose([
                             transforms.Select(['valence', 'arousal']),
                             transforms.Binary(3.0),
                             transforms.BinariesToCategory()
                         ]))

    print(dataset[0].shape)