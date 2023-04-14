from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class Trainer:
    def __init__(self,
                model: torch.nn.Module,
                device: torch.device,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                training_dataloader: Dataset,
                validation_dataloader:Optional[Dataset]=None,
                lr_scheduler:torch.optim.lr_scheduler=None,
                epochs: int=100,
                epoch: int=0,
                notebook: bool=False,
                ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1
            
            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_dataloader is not None:
                self._validate()
            
            """Learning rate schduler block"""
            if self.lr_scheduler is not None:
                if (self.validation_dataloader is not None and self.lr_scheduler.__class__.__name__ =="ReduceLROnPlateau"):
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.batch()
        return self.training_loss, self.validation_loss, self.learning_rate
    
    def _train(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
        self.model.train()
        train_losses = []
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False
        )

        for i,(x,y) in batch_iter:
            #send to device
            input_x, target_y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad() 
            out = self.model(input_x)
            loss = self.criterion(out, target_y)
            loss_value = self.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()
    
    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False
        )

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()



