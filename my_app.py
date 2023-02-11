import torch
import torchvision
import torchvision.transforms as transforms

from NeuralNetwork import model

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
NB_OUTPUTS = 10

class ai_colorizer():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(epochs=EPOCHS, batch_size=BATCH_SIZE, l_r=LR, nb_outputs=NB_OUTPUTS)
        self.model.to(self.device)
        self.train_loader = self.get_data("./datasets/")
        self.test_loader = self.get_data("./datasets/", train=False)

    def get_data(self, path, train: bool = True):
        transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels = 1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]
            )
        dataset = torchvision.datasets.CIFAR100(
            root=path,
            train=train,
            download=True,
            transform=transform
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
            )
        return dataloader

    def model_save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def model_load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


