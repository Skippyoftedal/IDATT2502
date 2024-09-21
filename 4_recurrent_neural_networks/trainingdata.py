from typing import List

import numpy as np
import torch
from numpy import ndarray
from torch import device


def get_x_train(emoji: dict[str, int], possible_letters: List[str],
    encodings: ndarray,
    dev: device, min_len: int):
    x_train = []  # Use a list to collect indices
    for word in emoji.keys():
        indices = []
        word_len = len(word)
        # Pad the word if it's shorter than min_len
        padded_word = word + ' ' * (min_len - word_len)

        for i in range(min_len):
            letter_index = possible_letters.index(padded_word[i])
            indices.append(letter_index)

        # Get the one-hot encoded representation for each letter
        encoded = encodings[indices]
        x_train.append(encoded)

    # Stack the list into a numpy array and convert to tensor
    x_train_np = np.stack(x_train)  # Stack the list into a numpy array

    # Convert to PyTorch tensor and move to the specified device
    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32).to(dev)

    # Reshape from (7, 4, 13) to (7, 4, 1, 13)
    return x_train_tensor.unsqueeze(2)




def get_y_train(emoji_dict: dict[str, int], dev, min_len):
    y_train = torch.tensor([
        [[emoji_value] for _ in range(min_len)]
        for emoji_value in emoji_dict.values()
    ], dtype=torch.float)

    return y_train.to(dev)
