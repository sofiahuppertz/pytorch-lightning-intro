# %%
import lightning as L
import torch

print("Lightning version:", L.__version__)
print("Torch version:", torch.__version__)
print("MPS is available:", torch.mps.is_available())

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# %%
L.seed_everything(378194)

# %%
class CIFAR10DataModule(L.LightningDataModule):
   
    def __init__(self, data_dir = "./data", batch_size = 64):
        # Define any custom user-defined parameters
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
    def prepare_data(self):
        # Download the data
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.cifar_train = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform_train
            )
            self.cifar_val = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )
        
        if stage == 'test' or stage is None:
            self.cifar_test = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=11, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=11, persistent_workers=True)


# %%
class CIFAR10CNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self .pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
       x, y = batch
       y_hat = self(x)
       loss = F.cross_entropy(y_hat, y)
       acc = (y_hat.argmax(1) == y).float().mean()
       self.log('val_loss', loss)
       self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
       x, y = batch
       y_hat = self(x)
       loss = F.cross_entropy(y_hat, y)
       acc = (y_hat.argmax(1) == y).float().mean()
       self.log('test_loss', loss)
       self.log('test_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }

# %%
checkpoint_callback = ModelCheckpoint(
   dirpath="checkpoints",
   monitor="val_loss",
   filename="cifar10-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
   save_top_k=3,
   mode="min",
    
)

# %%
logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name="cifar10_cnn"
)

wandb_logger = WandbLogger(
	project="cifar10_cnn",
)

# %%
early_stopping = EarlyStopping(
        monitor="val_loss", 
        patience=5, 
        mode="min",
        verbose=False
)

# %%
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

# create your own theme!
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="train",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)


# %%
dm = CIFAR10DataModule()

model = CIFAR10CNN()

trainer = L.Trainer(
    fast_dev_run=False,
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stopping, progress_bar],
    logger=wandb_logger,
    accelerator="mps" if torch.mps.is_available() else "cpu",
    devices="auto",
    enable_progress_bar=True,
)

# %%
import wandb
# Log in to your account
wandb.login(key="xxx")

# %%
trainer.fit(model, dm)

# %%
torch.save(model.state_dict(), "model")

# %%



