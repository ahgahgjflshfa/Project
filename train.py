import os.path
import torch
import argparse
from tqdm import trange
from torch import nn
from models.model import LSTMModel
from dataset import WeatherDataset
from torch.utils.data import DataLoader
from utils.plot_functions import plot_loss_curves
from utils.accuracy_function import accuracy

def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn,
               device: str = "cpu"):
    """
    Performs a single training step on the given model using the given training data and hyperparameters.
    Args:
        model: The model to train.
        data_loader: The DataLoader object used for training.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:

    """
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)

        # Calculate loss and accuracy (per batch)
        loss = loss_fn(y_logits.unsqueeze(2), y)
        train_loss += loss.item()   # Accumulate train loss
        train_acc += acc_fn(y_logits.unsqueeze(2), y)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              acc_fn,
              device: str = "cpu"):
    """
    Performs a single testing step on the given model using the given testing data and hyperparameters.

    Args:
        model: The model to test.
        data_loader: The DataLoader object used for testing.
        loss_fn: The loss function used for testing.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:

    """
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)

            # Calculate loss and accuracy (per batch)
            loss = loss_fn(y_logits.unsqueeze(2), y)
            test_loss += loss.item()
            test_acc += acc_fn(y_logits.unsqueeze(2), y)

        # Calculate average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc

def train(model: nn.Module,
          epochs: int,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          acc_fn,
          device: str = "cpu") -> dict[str, list[float]]:
    """
    Trains the specified neural network model using the given training data and hyperparameters.

    Args:
        model: The neural network model to be trained.
        epochs (optional): The number of epochs (iterations over the entire training dataset) to train the model.
                           Default is 200.
        train_dataloader: The DataLoader object used for training.
        test_dataloader: The DataLoader object used for testing.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:
        A dictionary containing all train loss, train accuracy, test loss and test accuracy while training the model.
    """
    results = {"train_loss": [],
               "test_loss": [],
               "train_acc": [],
               "test_acc": []}

    progress_bar = trange(epochs, desc="Epochs", position=0)

    for epoch in progress_bar:
        # Train step
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           acc_fn=acc_fn,
                                           device=device)

        # Test step
        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        acc_fn=acc_fn,
                                        device=device)

        scheduler.step(test_loss)

        if test_loss < 7:
            break

        if epoch % 10 == 0:
            # progress_bar.update(1)
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f} | Train accuracy: {train_acc} | Test accuracy: {test_acc}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)

    return results

if __name__ == "__main__":
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = WeatherDataset("./data/train", 20)
    test_dataset = WeatherDataset("./data/test", 20)

    training_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testing_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LSTMModel(input_size=22,
                      hidden_unit=128,
                      num_layers=2,
                      output_size=24)

    model.to(device)

    # Prepare optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1.5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Check if there is a saved model
    save_path = "models/model.pth"

    try:
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"Model parameters loaded from {save_path}")
    except RuntimeError as e:
        print(f"Saved model state dict miss match current model.")

    results = train(model=model,
                    epochs=10,
                    train_dataloader=training_dataloader,
                    test_dataloader=testing_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    acc_fn=accuracy,
                    device=device)

    plot_loss_curves(results)

    # Save model
    torch.save(model.state_dict(), save_path)

    print(f"Model parameters saved to {save_path}")