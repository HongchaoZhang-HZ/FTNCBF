import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class SensorDataClassifier(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(SensorDataClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define your neural network model here
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

        # Define your loss function (Binary Cross-Entropy) and optimizer
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Forward pass through the neural network
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        predictions = self(data)
        loss = self.loss_fn(predictions, labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return self.optimizer




# # Create dummy data and labels (replace with your actual data)
# n = 6
# m = 3
#
# X_train = torch.randn(8, n).to(torch.float)  # Ensure float data type
# y_train = torch.randint(0, 2, (8, m)).to(torch.float)  # Ensure float data type
#
# X_val = torch.randn(8, n).to(torch.float)  # Ensure float data type
# y_val = torch.randint(0, 2, (8, m)).to(torch.float)  # Ensure float data type
#
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
#
# # Set up TensorBoard logger
# logger = pl.loggers.TensorBoardLogger("logs", name="sensor_classifier")
#
# model = SensorDataClassifier(n, m)
# trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10, logger=logger)
#
# trainer.fit(model, train_loader, val_loader)
