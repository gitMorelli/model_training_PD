import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import os 

MODEL = 'resnet50'
OUTPUT_FOLDER = "/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist"
CHECKPOINT_PATH = os.path.join(OUTPUT_FOLDER, MODEL ,"checkpoints")
log_root = os.path.join(OUTPUT_FOLDER, 'tensor_board_logging')

NUM_EPOCHS = 50

mnist_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # upscale from 28×28
    transforms.Grayscale(num_output_channels=3),  # repeat channel ×3 → RGB
    transforms.ToTensor(),                  # → [0, 1] float tensor, shape (3, 224, 224)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # ImageNet stats — same mean/std
        std=[0.229, 0.224, 0.225]           # the pretrained model was trained with
    ),
])

base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST Mean and Std
        ])

# --- 1. MODEL DEFINITION ---
class MNISTResNet(L.LightningModule):
    def __init__(
        self,
        model: str = 'resnet18',
        lr_head: float = 1e-3,
        lr_late: float = 1e-4,
        lr_early: float = 1e-5,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model == 'resnet18': 
            self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model == 'resnet50':
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Adapt the final fully connected layer for 10 classes instead of 1000
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        param_groups = [
            # Early layers — preserve ImageNet features
            {"params": self.model.layer1.parameters(), "lr": self.hparams.lr_early},
            {"params": self.model.layer2.parameters(), "lr": self.hparams.lr_early},
            # Late layers — moderate adaptation
            {"params": self.model.layer3.parameters(), "lr": self.hparams.lr_late},
            {"params": self.model.layer4.parameters(), "lr": self.hparams.lr_late},
            # Head — trains most aggressively
            {"params": self.model.fc.parameters(),     "lr": self.hparams.lr_head},
        ]

        # Note: conv1, bn1 are excluded → frozen (no grad updates)
        # They learn the most generic low-level features (edges),
        # no benefit in touching them for MNIST

        optimizer = torch.optim.Adam(param_groups, weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,  # one cosine cycle over full training
            eta_min=1e-6,                   # floor LR — never fully zeroes out
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step the scheduler once per epoch
                "monitor": "val_loss", # not used by cosine, but good practice to include
            },
        }


# --- 2. DATA PIPELINE ---
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = mnist_transform

    def prepare_data(self):
        # Download datasets
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # Split train set into train and validation sets (55k / 5k split)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=8)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

# --- 3. TRAINING & EVALUATION LOOP ---
if __name__ == "__main__":
    args = get_args()
    num_workers = args.num_workers
    # Set random seed for reproducibility
    L.seed_everything(42)

    # Initialize data and model modules
    dm = MNISTDataModule(batch_size=128)
    model = MNISTResNet(model=MODEL,num_epochs=NUM_EPOCHS,lr_head=1e-3, lr_late=1e-4, lr_early=1e-5, weight_decay=1e-4)

    # Configure checkpoint callback to monitor and save the absolute best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=CHECKPOINT_PATH,
        filename="best-mnist-{epoch:02d}-{val_loss:.4f}"
    )

    # 3. Initialize the logger
    tb_logger = TensorBoardLogger(
        save_dir=log_root,
        name=MODEL,
        log_graph=True,
    )

    # Setup the Lightning Trainer
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="auto",  # Automatically uses GPU if available
        devices=1,
        logger = tb_logger,
    )

    # Execute training and validation loops
    trainer.fit(model, datamodule=dm)

    # --- MODEL EVALUATION LOOP ---
    # Automatically loads the best checkpoint found during training according to 'val_loss'
    print("\n" + "="*50)
    print("TRAINING COMPLETE. RUNNING EVALUATION LOOP ON THE BEST MODEL...")
    print("="*50)
    
    val_results = trainer.validate(model, datamodule=dm, ckpt_path="best")
    
    # Extract and display the best metrics cleanly
    best_val_loss = val_results[0]["val_loss"]
    best_val_acc = val_results[0]["val_acc"]

    print("\n" + "-"*30)
    print(f"Final Best Validation Loss:     {best_val_loss:.4f}")
    print(f"Final Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print("-"*30)