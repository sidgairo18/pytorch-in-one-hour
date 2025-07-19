import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# distributed training imports
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

# create toy dataset
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

def compute_accuracy(model, dataloader, device):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = predictions == labels
        correct += torch.sum(compare)
        total_examples += len(compare)


    return (correct / total_examples).item()

def prepare_dataset():
    # toy data
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])

    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    factor = 4
    X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    y_train = y_train.repeat(factor)
    X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    y_test = y_test.repeat(factor)

    # setup dataset
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # setup dataloaders
    train_loader = DataLoader(
            dataset = train_ds,
            batch_size = 2,
            shuffle = False,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
            sampler=DistributedSampler(train_ds),
            )
    test_loader = DataLoader(
            dataset = test_ds,
            batch_size = 2,
            shuffle = False,
            num_workers=0,
            )

    return train_loader, test_loader


def ddp_setup(rank, world_size):

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"
        init_process_group(backbend="gloo", rank = rank, world_size = world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    
    # initialize a new model object
    model = NeuralNetwork(2, 2)
    model.to(rank)
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()


        for batch_idx, (features, labels) in enumerate(train_loader):
            # Transfer the data onto the GPU.
            features, labels = features.to(rank), labels.to(rank)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")

        model.eval()

        if rank == 0:
    
            try:
                train_acc = compute_accuracy(model, train_loader, device=rank)
                print(f"[GPU{rank}] Training accuracy", train_acc)
                test_acc = compute_accuracy(model, test_loader, device=rank)
                print(f"[GPU{rank}] Test accuracy", test_acc)

            except ZeroDivisionError as e:
                raise ZeroDivisionError(
                        f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
                        "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
                        f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code in prepare_dataset()"
                        )

            # save model checkpoint
            torch.save(model.state_dict(), "model.pth")

    destroy_process_group()




if __name__ == '__main__':

    print("Is cuda available?", torch.cuda.is_available())

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    # Only print on rank 0 to avoid duplicate prints from each GPU process
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())


    # set manual seed for torch
    torch.manual_seed(123)
    num_epochs = 3
    main(rank, world_size, num_epochs)



    """ 
    # test model loading
    model = NeuralNetwork(2, 2) # needs to match the original model exactly
    msg = model.load_state_dict(torch.load("model.pth", weights_only=True))
    print("Model loaded!", msg)
    """
