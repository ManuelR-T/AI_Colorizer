import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, epochs, batch_size, l_r, nb_outputs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.l_r = l_r

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.reshape = nn.Flatten()
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, nb_outputs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.l_r)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.reshape(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def accuracy(self, y_tests, y_pred):
        correct = 0
        total = 0
        for idx, i in enumerate(y_pred):
            if torch.argmax(i) == y_tests[idx]:
                correct += 1
            total += 1
        return correct / total * 100

    def train(self, x, y):
        for epoch in range(self.epochs):
            for i, data in enumerate(x):
                inputs, y_tests = data
                self.optimizer.zero_grad()
                y_pred = self.forward(inputs)
                loss = self.loss(y_pred, y_tests)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    loss = loss.item()
                    accuracy = self.accuracy(y_tests, y_pred)
                    print(f"Epoch: {epoch + 1} - Loss: {loss}")
                    print(f"Epoch: {epoch + 1} - Accuracy: {accuracy}%")
