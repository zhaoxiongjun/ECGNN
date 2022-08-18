import argparse
import glob
import os
import time
import random
import json
import wandb
import numpy as np
from tqdm import tqdm
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from utils.metrics import Metrics
from models.IMLENet_GNN import ECGNN

from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.metrics import Metrics, AUC, metric_summary

from utils import ecg_data
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('--train', type=bool, default=True, help='train and valid')
parser.add_argument('--test', type=bool, default=True, help='test')
parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')

parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')

parser.add_argument("--model", type=str, default="ecgnn", help="Select the model to train")
parser.add_argument('--dataset', type=str, default='ptb', help='ptb/ICBEB')
parser.add_argument("--loggr", type=bool, default=True, help="Enable wandb logging")

args = parser.parse_args()

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)

def print_datainfo(dataset):
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}') 

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def one_hot(x, class_count):
	return torch.eye(class_count)[x,:]

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim,
    loss_func,
    train_loader,
    epoch: int,
    loggr: None,
    num_classes: int,
) -> Tuple[float, float, float]:
    """Training of the model for one epoch.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    optimizer: torch.optim
        Optimizer to be used.
    loss_func: torch.nn._Loss
        Loss function to be used.
    dataset: torch.utils.data.DataLoader
        Dataset to be used.
    epoch: int, optional
        The current epoch.
    device: torch.device
        Device to be used.
    loggr: bool, optional
        To log wandb metrics. (default: False)

    """

    model.train()

    pred_all = []
    loss_all = []
    gt_all = []

    for _, data in tqdm(enumerate(train_loader), desc="train"):
        data = data.to(device)
        pred = model(data)

        pred_all.append(pred.cpu().detach().numpy())
       # print(batch_y.type(), pred.type())
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        loss = loss_func(pred, y_true.to(device))
        loss_all.append(loss.cpu().detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gt_all.append(y_true.cpu().detach().numpy())

    print("Epoch: {0}".format(epoch))
   # print("Train loss: ", np.mean(loss_all))
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")

    if loggr is not None:
        loggr.log({"train_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"train_roc_score": roc_score, "epoch": epoch})
        loggr.log({"train_loss": np.mean(loss_all), "epoch": epoch})

    print(f'train_loss: {np.mean(loss_all)}, train_mean_accuracy: {mean_acc},train_roc_score: {roc_score}')

    return np.mean(loss_all), mean_acc, roc_score

def test_epoch(
    model: nn.Module,
    loss_func: torch.optim,
    loader,
    epoch: int,
    loggr: None,
    num_classes: int,
) -> Tuple[float, float, float]:
    """Testing of the model for one epoch.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    loss_func: torch.nn.BCEWithLogitsLoss
        Loss function to be used.
    dataset: torch.utils.data.DataLoader
        Dataset to be used.
    epoch: int, optional
        The current epoch.
    device: torch.device
        Device to be used.
    loggr: bool, optional
        To log wandb metrics. (default: False)

    """

    model.eval()

    pred_all = []
    loss_all = []
    gt_all = []

    for _, data in tqdm(enumerate(loader), desc="valid"):
        data = data.to(device)
        pred = model(data)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        loss = loss_func(pred, y_true.to(device))

        pred_all.append(pred.cpu().detach().numpy())
        gt_all.append(y_true.cpu().detach().numpy())
        loss_all.append(loss.cpu().detach().numpy())

    # print("Test loss: ", np.mean(loss_all))
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")

    if loggr is not None:
        loggr.log({"test_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"test_roc_score": roc_score, "epoch": epoch})
        loggr.log({"test_loss": np.mean(loss_all), "epoch": epoch})

    print(f'test_loss: {np.mean(loss_all)}, test_mean_accuracy: {mean_acc},test_roc_score: {roc_score}')

    return np.mean(loss_all), mean_acc, roc_score


def train(
    loggr,
    train_loader,
    val_loader,
    model: nn.Module,
    epochs: int = 60,
    name: str = "ecgnn",
    num_classes: int = 9,
) -> None:
    """Data preprocessing and training of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    epochs: int, optional
        Number of epochs. (default: 60)
    loggr: bool, optional
        To log wandb metrics. (default: False)
    name: str, optional
        Name of the model. (default: 'imle_net')

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    loss_func = torch.nn.BCEWithLogitsLoss()  # with sigmoid

    best_score = 0.0
    for epoch in range(epochs):

        train_results = train_epoch(model, optimizer, loss_func, train_loader, epoch, loggr=loggr, num_classes=num_classes)

        test_results = test_epoch(model, loss_func, val_loader, epoch, loggr=loggr, num_classes=num_classes)

        scheduler.step()

        if epoch >= 3 and best_score <= test_results[2]:
            best_score = test_results[2]
            save_path = os.path.join(os.getcwd(), "checkpoints/", f"{name}_weights.pt")
            torch.save(model.state_dict(), save_path)
          #  dump_logs(train_results, test_results, name)


def test(
    model: nn.Module,
    test_loader,
    num_classes
)-> None:

    pred_all = []
    loss_all = []
    gt_all = []

    for i, data in tqdm(enumerate(test_loader), desc="test"):
        data = data.to(device)
        pred = model(data)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        pred_all.append(pred.cpu().detach().numpy())
        
        gt_all.append(y_true.cpu().detach().numpy())

    
    pred_all = np.concatenate(pred_all, axis=0)
    y_test = np.array(np.concatenate(gt_all, axis=0))
    roc_score = roc_auc_score(y_test, pred_all, average="macro")
    acc, mean_acc = Metrics(y_test, pred_all)
    class_auc = AUC(y_test, pred_all)
    summary = metric_summary(y_test, pred_all)

    # ecg_data.challenge_metrics(y_test, one_hot(np.argmax(pred_all, axis=1), num_classes))
    print(f"class wise accuracy: {acc}")
    print(f"accuracy: {mean_acc}")
    print(f"roc_score : {roc_score}")
    print(f"class wise AUC : {class_auc}")
    print(f"class wise precision, recall, f1 score : {summary}")
    

    logs = dict()
    logs["roc_score"] = roc_score.tolist()
    logs["mean_acc"] = mean_acc
    logs["accuracy"] = acc
    logs["class_auc"] = class_auc
    logs["class_precision_recall_f1"] = summary

    name = "output"
    logs_path = os.path.join(os.getcwd(), "logs", f"{name}_logs.json")
    jsObj = json.dumps(logs)
    fileObject = open(logs_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()


if __name__ == "__main__":

    data_root = "./data/"
    sampling_rate = 100
    # Load raw ecg data
    data_path = os.path.join(data_root, args.dataset, "raw/")
    _, _,Y = ecg_data.load_dataset(data_path, sampling_rate)

    # Build graph datasets
    ecg_dataset = ecg_data.ECGDataset(os.path.join(data_root, args.dataset))
    print_datainfo(ecg_dataset)
    args.num_classes = 5
    args.num_features = ecg_dataset.num_features
  
    # Split dataset
    train_dataset, val_dataset, test_dataset = ecg_data.select_dataset(ecg_dataset, Y)
    # Standardization
    # train_dataset, val_dataset, test_dataset = ecg_data.data_scaler(train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build Model
    model = ECGNN(args).double().to(device)

    # Logging
    args.logger = None
    if args.loggr:
        wandb = wandb.init(
            project="IMLE-Net",
            name=args.model,
            notes=f"Model: {args.model} with batch size: {args.batch_size} and epochs: {args.epochs}",
            save_code=True,
        )
        args.logger = wandb

    # Model training 
    if args.train:
        print("<=============== Start Training ===============>")
        train(
            loggr=args.logger,
            model=model,
            epochs=args.epochs,
            name=args.model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_classes=args.num_classes,
        )

    # Model testing
    if args.test:
        print("<=============== Start Testing ===============>")
        path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights.pt")
        model.load_state_dict(torch.load(path_weights))

        test(model, test_loader, num_classes=args.num_classes)


    



