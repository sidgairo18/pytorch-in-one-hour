import torch
import torch.nn as nn
import torch.nn.functional as F

# defining a multilayer neural network class with torch.nn.Module

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # output layer
                torch.nn.Linear(20, num_outputs)
                )

    def forward(self, x):
        logits = self.layers(x)
        return logits

if __name__ == '__main__':

    # initialize a new model object
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)

    print(model)

    # printing the number of trainable paramaters in the model we instantiated
    num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
            )

    print("Total number of trainable model parameters: ", num_params)

    # print first layer weight matrix
    print(model.layers[0].weight, model.layers[0].weight.shape)
    print(model.layers[0].bias, model.layers[0].bias.shape)

    X = torch.rand((1, 50))
    with torch.no_grad():
        out_logits = model(X)
        out_probs = torch.softwax(out_logits, dim=1)
    print(out)


