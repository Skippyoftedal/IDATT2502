from typing import List

import numpy as np
import torch
import torch.nn as nn

from utility.deviceUtility import get_best_available_device


HIDDEN_SIZE = 128

class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, d):
        self.device = d
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, HIDDEN_SIZE, device=self.device)  # HIDDEN_SIZE is the state size
        self.dense = nn.Linear(HIDDEN_SIZE, encoding_size, device=self.device)  # HIDDEN_SIZE is the state size

        self.hidden_state = None
        self.cell_state = None

    def reset(self, batch_size=1):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, HIDDEN_SIZE).to(self.device)  # Shape: (num layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, HIDDEN_SIZE))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y_):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y_.argmax(1))


def get_x_train(word: str, possible_letters: List[str], encodings, dev):
    indices = [possible_letters.index(letter) for letter in word]
    return torch.tensor(encodings[indices]).view(len(indices), 1, -1).to(dev)


def get_y_train(word: str, possible_letters: List[str], encodings, dev):
    indices = [possible_letters.index(letter) for letter in word]
    return torch.tensor(encodings[indices]).to(dev)


if __name__ == '__main__':
    calc_device = get_best_available_device()

    possible_letter_sentence = " hello world"
    index_to_char = list(set(possible_letter_sentence))
    print(f"(Index -> char) map is {index_to_char}")

    letters = len(index_to_char)

    char_encodings = np.eye(letters, dtype=np.float32)

    print(char_encodings[[0, 1, 2, 3, 3, 4]])
    print("")
    print(char_encodings[[1, 2, 3, 3, 4, 0]])


    x_train = get_x_train(" hello world", index_to_char, char_encodings, calc_device) #' ' -> h, 'h' -> e osv?
    y_train = get_y_train("hello world ", index_to_char, char_encodings, calc_device)

    print(f"x train: {x_train}")
    print(f"y train: {y_train}")
    print(f"x shape is: {x_train.shape}")
    print(f"y shape is: {y_train.shape}")

    model = LongShortTermMemoryModel(len(char_encodings), calc_device)

    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
    for epoch in range(10000):
        model.reset(batch_size=x_train.size(1))  # Ensure the batch size is set
        optimizer.zero_grad()
        loss = model.loss(x_train, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.reset(batch_size=1)
            text = ' '

            input_tensor = torch.tensor([[char_encodings[0]]], dtype=torch.float32).to(calc_device)
            for _ in range(50):
                y = model.f(input_tensor)
                next_char_idx = y.argmax(1).item()
                text += index_to_char[next_char_idx]
                input_tensor = torch.tensor([[char_encodings[next_char_idx]]], dtype=torch.float32).to(calc_device)

            print(f'Epoch {epoch}: {text}')
