from pathlib import Path

import torchmetrics
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm


def eval_model(model: Module,
               data_loader: DataLoader,
               loss_fn: Module,
               accuracy_fn: torchmetrics.Metric,
               device: torch.device):
    torch.manual_seed(42)
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model.forward(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(torch.argmax(y_pred, dim=-1), y)
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, "model_loss": loss.item(), "model_acc": acc.item()}


def train_on_batches(model: Module,
                     data_loader: DataLoader,
                     loss_fn: Module,
                     optimizer: torch.optim,
                     accuracy_fn: torchmetrics.Metric,
                     device: torch.device):
    torch.manual_seed(42)
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model.forward(X)  # Forward pass
        train_loss = loss_fn(y_pred, y)  # Calc loss per batch
        train_loss += train_loss  # accumulate loss with loss from previous batches
        train_acc += accuracy_fn(torch.argmax(y_pred, dim=-1), y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()  # updating model parameters once per batch

    train_loss /= len(data_loader)  # average loss value across all batches in dataloader
    train_acc /= len(data_loader)  # average accuracy value across all batches in dataloader
    print(f"\nTrain loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")


def test_on_batches(model: Module,
                    data_loader: DataLoader,
                    loss_fn: Module,
                    accuracy_fn: torchmetrics.Metric,
                    device: torch.device):
    torch.manual_seed(42)
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model.forward(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(torch.argmax(test_pred, dim=-1), y)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def save_model(model: Module,
               model_name: str):
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {model_path}")
    torch.save(obj=model.state_dict(),  # saving the state_dict() only saves the models learned parameters
               f=model_path / model_name)


def load_model(model: Module,
               model_name: str):
    model_path = Path("models")
    print(f"Loading model from: {model_path}/{model_name}")
    model.load_state_dict(torch.load(f=model_path / model_name))


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device):
    pred_labels = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for image in data:
            image = torch.unsqueeze(image, dim=0).to(device)  # add batch dimension
            pred_logit = model.forward(image)  # output is raw logit
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)  # logit -> prediction probability
            pred_label = pred_prob.argmax()
            pred_labels.append(pred_label.cpu())  # get pred_prob to cpu
    return torch.stack(pred_labels)
