import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class ExampleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == '__main__':
    # Initialize a PyTorch Lightning trainer
    trainer = pl.Trainer(
        devices=2,
        num_nodes=2,
        accelerator='gpu',
        max_epochs=10,
        ip_address='127.0.0.1',  # or 'localhost' or your IP address
        port=0,  # set to 0 to use a random port
        address_family='ipv4',  # set the IP version explicitly

        # logger=pl.loggers.TensorBoardLogger('logs/')
    )

    # Initialize the example model
    model = ExampleModel()

    # Train the model
    trainer.fit(model)
