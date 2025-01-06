"""Module providing GNN training"""
import time

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader

from data_loading import LoadData
from models import SAGE
from params import args
from util import fix_seed, accuracy


fix_seed(args.seed)
cuda_use = not args.no_cuda and torch.cuda.is_available()
if cuda_use:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
device = "cuda" if torch.cuda.is_available() and cuda_use else "cpu"
print(device)

load = LoadData(args)
data_list_train = load.train_dataset()
data_list_test = load.test_dataset()
in_num = len(data_list_train[0].x[1])
out_num = len({data_list_train[i].y for i in range(0, len(data_list_train))})

train_loader = DataLoader(data_list_train, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(data_list_test, batch_size=args.batch)

model = SAGE(in_num, out_num).to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = OneCycleLR(optimizer,
                       max_lr=0.01,
                       steps_per_epoch=len(train_loader),
                       epochs=args.epochs)

criterion = torch.nn.CrossEntropyLoss()
history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}

for step, t_data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {t_data.num_graphs}')
    print(t_data)
    print()


def train(epoch):
    '''
    GNN training
    '''
    model.train()
    train_loss = 0.0
    time_now = time.time()
    for _, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        acc = accuracy(outputs, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()

    scheduler.step()

    print(f'Epoch:{epoch+1:04d},', end=" ")
    print(f'loss_train:{loss.item():.4f},', end=" ")
    print(f'acc_train: {acc.item():.4f},', end=" ")
    print(f'time: {time.time() - time_now:.4f}s')


def test():
    '''
    GNN testing
    '''
    model.eval()
    time_now = time.time()
    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            loss += criterion(outputs, data.y)
            _, predicted = torch.max(outputs, 1)
            total += data.y.size(0)
            batch_num += 1
            correct += (predicted == data.y).sum().cpu().item()

    print('\n###### Test Results ######')
    print(f'loss_test: {loss:.4f},', end=" ")
    print(f'acc_test: {correct/total:.4f},', end=" ")
    print(f'time: {time.time() - time_now:.4f}s')


if __name__ == '__main__':
    for each_epoch in range(args.epochs):
        train(each_epoch)

    test()
