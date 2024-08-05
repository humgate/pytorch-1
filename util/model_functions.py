from pathlib import Path

import torchmetrics
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm

from util.timer import Timer


def eval_model(model: Module,
               data_loader: DataLoader,
               loss_fn: Module,
               accuracy_fn: torchmetrics.Metric,
               device: torch.device):
    torch.manual_seed(42)
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in tqdm(data_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(torch.argmax(y_pred, dim=-1), y)
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, "model_loss": loss.item(), "model_acc": acc.item()}


def train_step_on_batches(model: Module,
                          data_loader: DataLoader,
                          loss_fn: Module,
                          optimizer: torch.optim,
                          accuracy_fn: torchmetrics.Metric,
                          device: torch.device):

    train_loss = 0
    model.train()

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model.forward(x)  # Forward pass
        loss = loss_fn(y_pred, y)  # Calc loss per batch
        train_loss += loss.item()  # accumulate loss with loss from previous batches
        accuracy_fn(torch.argmax(y_pred, dim=-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # updating model parameters once per batch

    train_loss /= len(data_loader)  # average loss value across all batches in dataloader
    train_acc = accuracy_fn.compute().item()  # average accuracy value across all batches in dataloader

    return train_loss, train_acc


def test_step_on_batches(model: Module,
                         data_loader: DataLoader,
                         loss_fn: Module,
                         accuracy_fn: torchmetrics.Metric,
                         device: torch.device):

    test_loss = 0
    model.eval()

    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            test_pred = model.forward(x)
            test_loss += loss_fn(test_pred, y).item()
            accuracy_fn(torch.argmax(test_pred, dim=-1), y)
        test_loss /= len(data_loader)
        test_acc = accuracy_fn.compute().item()

    return test_loss, test_acc


def train(model: Module,
          train_data_loader: DataLoader,
          test_data_loader: DataLoader,
          loss_fn: Module,
          optimizer: torch.optim,
          accuracy_fn: torchmetrics.Metric,
          epochs: int,
          device: torch.device):

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    timer = Timer()

    timer.start_timer()
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = (
            train_step_on_batches(model, train_data_loader, loss_fn, optimizer, accuracy_fn, torch.device(device)))
        test_loss, test_acc = (
            test_step_on_batches(model, test_data_loader, loss_fn, accuracy_fn, torch.device(device)))
        print(f"\nEpoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}"
              f" | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        # As we are using torchmetrics.Accuracy() and in train_step_on_batches and test_step_on_batches methods
        # the metric keeps accumulating the correct prediction counts, we need to reset the metric at the end of each
        # epoch.
        accuracy_fn.reset()

    timer.stop_timer()
    timer.print_elapsed_time()
    del model
    torch.cuda.empty_cache()

    return results


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
        for image in tqdm(data, desc="Making predictions..."):
            # add batch dimension because our model Flatten layer flattens tensor starting at 1st dimension
            # so if we do not add 0th dimension, flattened matrix will have wrong size to be multiplied to output
            image = torch.unsqueeze(image, dim=0).to(device)
            pred_logit = model.forward(image)  # output is raw logit
            # pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)  # logit -> prediction probability
            # pred_label = pred_prob.argmax()
            pred_label = pred_logit.argmax()  # argmax the logit directly if we do not need the probability itself
            pred_labels.append(pred_label.cpu())  # get pred_prob to cpu
    return torch.stack(pred_labels)
