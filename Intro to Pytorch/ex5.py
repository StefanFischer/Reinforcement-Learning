"""PyTorch MNIST Example."""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    """A neural network implementation."""
    def __init__(self):
        super(Net, self).__init__()
        # Define the member variables for your layers.
        # Use the appropriate layers from torch.nn
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fullyConnected1 = nn.Linear(in_features=2304, out_features=128)
        self.act3 = nn.ReLU()
        self.fullyConnected2 = nn.Linear(in_features=128, out_features=10)
        self.act4 = nn.Softmax()


    def forward(self, x):
        # Implement one forward pass of your neural network.
        conv1_output = self.conv1(x)
        act1_output = self.act1(conv1_output)
        conv2_output = self.conv2(act1_output)
        act2_output = self.act2(conv2_output)
        flatten_output = self.flatten(act2_output)
        fc1_output = self.fullyConnected1(flatten_output)
        act3_output = self.act3(fc1_output)
        fc2_output = self.fullyConnected2(act3_output)
        act4_output = self.act4(fc2_output)
        return act4_output


def train(model, device, train_loader, optimizer, epoch, args):
    """Train the model for one epoch."""
    # This indicates to the model that it is used for training.
    # Will, e.g., change how dropout layers operate.
    model.train()

    # Inner training loop: Iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target to the correct device (cpu/gpu).
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        #  implement one step of the optimization:
        # * Calculate predictions
        predictions = model(data)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)
        # * Calculate the loss
        loss = F.binary_cross_entropy(input=predictions, target=one_hot_target.float())

        # * Backpropagate the loss to find the gradients
        loss.backward()


        # * Take one gradient step with your optimizer
        optimizer.step()

        # Assign your loss value here for easy printing
        loss_for_printing = loss
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_for_printing))
            if args["dry_run"]:
                break


def test(model, device, test_loader):
    """Test the model on the specified test set, and print test loss and accuracy."""
    # Similar to .train() above, this will tell the model it is used for inference.
    model.eval()

    # Accumulator for the loss over the test dataset
    test_loss = 0
    # Accumulator for the number of correctly classified items
    correct = 0

    # This block will not compute any gradients
    with torch.no_grad():
        # Similar to the inner training loop, only over the test_loader
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # TODO: Implement the same loss calculations as in training
            # No optimizer step here.
            predictions = model(data)
            one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)
            # * Calculate the loss
            test_loss += F.binary_cross_entropy(input=predictions, target=one_hot_target.float())

            # TODO: Calculate the predictions of your model over the batch


            # TODO: Calculate how many predictions were correct, and add them here
            predsFloat = torch.FloatTensor(predictions.cpu())
            predsToOne = torch.where(predsFloat >= 0.5, 1, 2)
            cnt = torch.where(predsToOne == one_hot_target.cpu(), 1, 0)
            correct += torch.sum(cnt)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_mnist_loaders(batch_size):
    """Creates train- and test-DataLoaders for MNIST with the specified batch size."""
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False, download=True,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader

def main():

    # Seed your model for reproducibility
    torch.manual_seed(4711)

    # If possible, use CUDA (i.e., your GPU) for computations.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Training Parameters
    learning_rate = 0.001
    batch_size = 30
    epochs = 14
    training_args = dict(
        log_interval=10,
        dry_run=False
    )

    # Retrieve DataLoaders for the train- and test-dataset.
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Create your network, and move it to the specified device
    model = Net().to(device)

    # Create your optimizer here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # The outer training loop (over epochs)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, training_args)
        test(model, device, test_loader)

    # Save the trained model.
    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()