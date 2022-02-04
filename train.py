import torch
import os
from basic_model import Net
from colorize_data import ColorizeData
from torch.utils.data import Dataset, DataLoader
import argparse

BATCHSIZE = 100
LEARNRATE = 0.001
MAXEPOCHS = 100

class Trainer:
    def __init__(self):
        # Define hparams here or load them from a config file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        # dataloaders
        train_dataset = ColorizeData('train')
        train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
        val_dataset = ColorizeData('verify')
        val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True)
        # Model
        model = Net().to(self.device)
        # Loss function to use
        criterion = torch.nn.MSELoss()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNRATE)
        # train loop
        lossList = [10]
        model.train()
        for epoch in range(MAXEPOCHS):
            lossTrain = 0
            for data, y in train_dataloader:
                optimizer.zero_grad()
                data.to(self.device)
                y.to(self.device)
                pred = model.forward(data)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                lossTrain += loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            lossVal = self.validate(model, val_dataloader)
            print(f'Epoch: {epoch+1:2} Train Loss: {lossTrain.item():10.8f} Validate Loss: {lossVal.item():10.8f}')
            lossList.append(lossVal.item())
            if (lossList[-1] < min(lossList[:-1])):
                torch.save(model.state_dict(), 'model_train.pth')
                print(f'save model, epoch: {epoch+1:2}')

        return model

    def validate(self, model, loader):
        # Validation loop begin
        model.to(self.device)
        model.eval()

        lo = 0
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            for data, y in loader:
                data.to(self.device)
                y.to(self.device)
                pred = model.forward(data)
                loss = criterion(pred, y)
                lo += loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Validation loop end
        # Determine your evaluation metrics on the validation dataset.
        return lo

def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='BATCHSIZE', help='the batch size', type=int)
    parser.add_argument('-lr', metavar='LEARNRATE', help='the learning rate', type=float)
    parser.add_argument('-m', metavar='MAXEPOCHS', help='the maxepochs', type=int)
    args = parser.parse_args()
    global BATCHSIZE
    global LEARNRATE
    global MAXEPOCHS
    if args.b: BATCHSIZE = args.b
    if args.lr: LEARNRATE = args.lr
    if args.m: MAXEPOCHS = args.m

    return parser

if __name__ == '__main__':
    ArgParser()
    print(f'b = {BATCHSIZE}, lr = {LEARNRATE}, max={MAXEPOCHS}')
    ins = Trainer()
    ins.train()