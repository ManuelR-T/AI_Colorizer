import torch
import torchvision
import torchvision.transforms as transforms
import gradio as gr

from NeuralNetwork import model

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
NB_OUTPUTS = 10
PATH = './datasets'

class AIColorizer():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(epochs=EPOCHS, batch_size=BATCH_SIZE, l_r=LR, nb_outputs=NB_OUTPUTS)
        self.model.to(self.device)
        self.train_loader_g =   self.get_data("./datasets/") # Train grayscale
        self.test_loader_g  =   self.get_data("./datasets/", train=False) # Test grayscale
        self.train_loader_c =   self.get_data("./datasets/", grayscale=False) # Train colored
        self.test_loader_c  =   self.get_data("./datasets/", grayscale=False, train=False) # Test colored

    def get_data(
            self,
            path,
            grayscale: bool = True,
            train: bool = True
            ):
        if grayscale:
            transform = transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels = 1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]
                )
        else:
            transform = transforms.Compose(
                    [
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

    def gradio_app(self):
        gr.Interface(
            fn=self.model.predict,
            inputs=gr.inputs.Image(shape=(32, 32)),
            outputs=gr.outputs.Image(shape=(32, 32))
        ).launch()

def train(ai_app):
    ai_app.model_load("./models/model.ia")
    response = input("Do you want to continue? (yes/no) ")
    if response.lower() == "yes" or response.lower() == "y":
        print("Saving model...")
        ai_app.model_save("./models/model.ia")
        print("Model saved!")

def gradio_app(ai_app):
    ai_app.model_load("./models/model.ia")
    ai_app.gradio_app()

def main():
    ai_app = ai_colorizer()
    while True:
        response = input("Enter 'train' to train the model or 'gradio' to launch the app: ")
        if response.lower() == "train" or response.lower() == "t":
            train(ai_app)
        elif response.lower() == "gradio" or response.lower() == "g":
            gradio_app(ai_app)
        else:
            print("Invalid input!")
            continue
        break

if __name__ == "__main__":
    main()