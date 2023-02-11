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

    def train(self, train_loader_g, train_loader_c):
        for epoch in range(self.epochs):
            for data_g, data_c in zip(train_loader_g, train_loader_c):
                inputs_g, labels_g = data_g
                inputs_c, labels_c = data_c

                self.optimizer.zero_grad()
                # output grayscale
                outputs_g   =   self.forward(inputs_g)
                loss_g      =   self.loss(outputs_g, labels_g)

                # output color
                outputs_c   =   self.forward(inputs_c)
                loss_c      =   self.loss(outputs_c, labels_c)

                # total loss
                loss_tot = loss_g + loss_c

                loss_tot.backward()
                self.optimizer.step()

            # accuracy and loss asserting
            loss = loss.item()
            accuracy_g = self.accuracy(outputs_g, labels_g)
            accuracy_c = self.accuracy(outputs_c, labels_c)
            print(f"Epoch: {epoch + 1} - Loss: {loss}")
            print(f"Epoch: {epoch + 1} - Accuracy: {accuracy_g}%")
            print(f"Epoch: {epoch + 1} - Accuracy: {accuracy_c}%")

