import torch
from unet import UNet
from trainer import Trainer
from dataloader import dataloader_training, dataloader_validation
from lr_rate_finder import LearningRateFinder

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

#criterion
criterion = torch.nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#trainer
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_dataloader=dataloader_training,
    validation_dataloader=dataloader_validation,
    lr_scheduler=None,
    epochs=2, epoch=0, notebook=True
)

if __name__ == '__main__':
    #learning rate finder
    lrf = LearningRateFinder(model, criterion, optimizer, device)
    lrf.fit(dataloader_training, steps=100)
    lrf.plot()

    #start training
    # training_loss, validation_loss, lr_rates = trainer.run_trainer()