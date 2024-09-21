import torch.nn as nn
import torch


class EmojiModel(nn.Module):
    def __init__(self, word_encoding_size, emoji_encoding_size, dev,
        hidden_s=128):
        self.device = dev
        self.hidden_size = hidden_s

        super(EmojiModel, self).__init__()

        self.lstm = nn.LSTM(word_encoding_size,
                            self.hidden_size,
                            device=self.device)

        self.dense = nn.Linear(self.hidden_size,
                               emoji_encoding_size,
                               device=self.device)

        self.hidden_state = None
        self.cell_state = None
        self.layers = 1
        self.batch_size = 1

    def reset(self):
        zero_state = torch.zeros(self.layers, self.batch_size,
                                 self.hidden_size).to(
            self.device)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self,
        x):

        out, (self.hidden_state, self.cell_state) = self.lstm(x, (
            self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, self.hidden_size))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y_):
        return nn.functional.cross_entropy(self.logits(x), y_.argmax(1))

    def train_model(self, x_train, y_train):
        print("Starting training")
        optimizer = torch.optim.RMSprop(self.parameters(), 0.001)  # 0.001
        for epoch in range(100):
            print(f"epoch {epoch}")
            for i in (range(x_train.size()[0])):
                self.reset()
                l = self.loss(x_train[i], y_train[i])
                if i == 0 and epoch % 100 == 0:
                    print(f"Loss: {l}")
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
