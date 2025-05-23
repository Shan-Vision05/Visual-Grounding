import torch
import torch.nn as nn
from utils.Util import *

class VisualGroundingTrainer():
    def __init__(self, model, device, train_dataloader, test_dataloader):

        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4
        )
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train_step(self):
        self.model.train()
        train_loss = 0
        train_acc = 0
        for batch, (X_Img, X_Text, y_bbox) in enumerate(self.train_dataloader):
            X_Img, X_Text, y_bbox = X_Img.to(self.device), X_Text.to(self.device), y_bbox.to(self.device)
            # batch[0], batch[1]['input_ids'], batch[1]['attention_mask']

            roi, y_pred = self.model(X_Img, X_Text['input_ids'], X_Text['attention_mask'])

            # plot_region_with_text(X_Img, X_Text['input_ids'], y_bbox, roi)
            # plot_region_with_text(X_Img, X_Text['input_ids'], y_pred.squeeze(dim=-1).argmax(dim=1), roi, predict=True)

            y = CreateBatchLabels(roi, y_bbox).to(self.device)
            loss = self.loss_fn(y_pred.squeeze(dim=-1), y.type(torch.long))

            train_loss += loss
            # print(train_loss)
            train_acc += self.accuracy_fn(y, y_pred.squeeze(dim=-1).argmax(dim=1))

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            # print(f'Batch {batch}')

        # self.scheduler.step()
        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        return train_loss, train_acc

    def eval_step(self):
        self.model.eval()

        with torch.inference_mode():
            test_loss = 0
            test_acc = 0
            for batch, (X_Img, X_Text, y_bbox) in enumerate(self.test_dataloader):
                X_Img, X_Text, y_bbox = X_Img.to(self.device), X_Text.to(self.device), y_bbox.to(self.device)

                roi, y_pred = self.model(X_Img, X_Text['input_ids'], X_Text['attention_mask'])
                y = CreateBatchLabels(roi, y_bbox).to(self.device)

                loss = self.loss_fn(y_pred.squeeze(dim=-1), y.type(torch.long))

                test_loss += loss
                # print(test_loss)
                test_acc +=  self.accuracy_fn(y, y_pred.squeeze(dim=-1).argmax(dim=1))
            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)
            return test_loss, test_acc