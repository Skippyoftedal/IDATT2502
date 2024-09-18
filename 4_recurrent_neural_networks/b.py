from typing import List

import numpy as np
import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

        self.hidden_state = None
        self.cell_state = None

    def reset(self, batch_size=1, device='cpu'):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128, device=device)  # Shape: (num layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


def get_x_train(word: str, possible_letters: List[str], encodings):
    indices = []
    for letter in word:
        if letter in possible_letters:
            i = possible_letters.index(letter)
            indices.append(i)
    return torch.tensor(encodings[indices]).view(len(indices), 1, -1)


def get_y_train(word: str, possible_letters: List[str], encodings):
    indices = []
    for letter in word:
        if letter in possible_letters:
            i = possible_letters.index(letter)
            indices.append(i)
    return torch.tensor(encodings[indices])


index_to_char = [' ',
                 '🎩',
                 'e',
                 'l',
                 'o',
                 'w',
                 'r',
                 'd']  # Example vocabulary
letters = len(index_to_char)

char_encodings = np.eye(letters, dtype=np.float32)

print(char_encodings[[0, 1, 2, 3, 3, 4]])
print("")
print(char_encodings[[1, 2, 3, 3, 4, 0]])

# x_train = torch.tensor([[char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]],
#                         [char_encodings[4]]])  # ' hello'
# y_train = torch.tensor([char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0]])  # 'hello '
# # print(f"x train: {x_train}")
# # print(f"y train: {y_train}")
# print(f"shape should: {x_train.shape}")
# print(f"shape should: {y_train.shape}")


x_train = get_x_train(" h           ", index_to_char, char_encodings)
y_train = get_y_train(" hello world ", index_to_char, char_encodings)

print(f"x train: {x_train}")
print(f"y train: {y_train}")
print(f"x shape is: {x_train.shape}")
print(f"y shape is: {y_train.shape}")

model = LongShortTermMemoryModel(len(char_encodings))

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(300):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 3 == 0:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]]))
        y = model.f(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(text)
