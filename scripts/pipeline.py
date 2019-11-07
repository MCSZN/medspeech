from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils import to_categorical
from typing import Callable
from numba import njit
import numpy as np
import librosa
import typing

DataFrame = typing.Generic

@njit
def shorten(audio: np.array, reduce_by: float = 0.33) -> np.array:
    """
    Shortens audio file by given rate
    :audio: wavfile array to be processed
    :reduce_by: rate by which to reduce
    """
    final_len = int(len(audio) * (1 - reduce_by))
    new = np.zeros(final_len)
    ratio = 1 / (1 - reduce_by)
    for i in range(final_len):
        new[i] = (audio[int(i * ratio)] + audio[int(i * ratio) + 1]) / ratio
    return new


def get_data(file: str, signal: Callable, length: int = 234500) -> np.array:
    """
    Load the file  into memory with Librosa
    Uses sample rate of 16k
    Pads the data up to 234.5k data points
    :param file: str path to file
    :param signal: function to apply signal processing
    :return: np.array of audio
    """
    data, _ = librosa.load(file, sr=16000)
    data = shorten(data)
    data = np.pad(data, (0, length - len(data)), "constant").astype(np.float32)
    data = signal(data)
    data = data[..., np.newaxis]
    return data


class Dataloader(Sequence):
    def __init__(
        self, dataset: DataFrame, set_: str, batch_size: int, signal: Callable, script: bool= False) -> None:
        self.dataset = dataset
        self.dataset.index = np.arange(self.dataset.shape[0])
        self.batch_size = batch_size
        self.signal = signal
        self.set_ = set_
        self.script = script
        self.path = "../data/recordings" if self.script else "data/recordings"

    def __len__(self) -> int:
        return int(np.ceil(self.dataset.shape[0] / self.batch_size))

    def __getitem__(self, idx: int) -> (np.array, np.array):
        data = self.dataset.loc[idx * self.batch_size : (idx + 1) * self.batch_size - 1]
        files = data["file_name"].values
        data = np.array(
            [
                get_data(
                    "{}/{}/{}".format(self.path,self.set_, file), self.signal
                )
                for file in files
            ]
        )
        target = np.array(
            self.dataset.loc[
                idx * self.batch_size : (idx + 1) * self.batch_size - 1
            ].prompt.values
        )
        target = to_categorical(target, num_classes=25)
        return data, target


def make_dl(dataset: DataFrame, set_: str, bs: int, signal: Callable) -> Dataloader:
    """
    Avoid redundancy on code
    Sub-samples dataset with training, testing or val set
    Loads the returned dataset into dataloader and returns the dataloader
    :param dataset: a pandas dataframe that is well clean
    :param set_: either 'train', 'test' or 'validate'
    :param bs: an int
    :param signal: a signal processing function
    :return: Dataloader with given parameters
    """
    data = dataset[dataset.set == set_]
    return Dataloader(data, set_, bs, signal)


if __name__ == "__main__":
    print("This file is made to be imported")
    print("It exports data loaders")
