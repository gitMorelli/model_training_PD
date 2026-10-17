from torch import nn
import torch
import torch.nn.functional as F

class SimpleMockModel(nn.Module):
    def __init__(self,seq_length=13*3):
        super().__init__()
        self.conv = nn.Conv2d(seq_length*3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10) # 10 output classes
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: [Batch, 6, 3, H, W]
        batch_size = x.shape[0]
        channels = x.shape[2] # Extract the channel dimension (3)
        # Flatten the time and channel dimensions: [Batch, 18, H, W]
        x = x.view(batch_size, -1, x.shape[3], x.shape[4]) 
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        return self.fc(x) 
class CustomBinaryCNN(nn.Module):
    def __init__(self):
        super(CustomBinaryCNN, self).__init__()
        
        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier Head
        self.flatten = nn.Flatten()
        # 128 channels * 14 * 14 spatial dimensions
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Crucial for small datasets
            nn.Linear(512, 2) # 2 output neurons
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
class FiveStageResidualStridedConvNet(nn.Module): #can get any input size -> consider using diff size for diff modalities
    """
    ~2.5M-parameter encoder producing a `feat_dim`-dimensional embedding.
 
    Args:
        feat_dim: dimension of the output feature vector.
        widths: channel width of each of the five stages. Halve them
            (16, 32, 64, 128, 128) for a ~0.7M-parameter version if you
            observe overfitting during fine-tuning.
        in_channels: number of input channels (1 for grayscale).
        dropout: dropout applied before the final linear head.
        ssl_projection: if True, also build a SimCLR-style MLP projection
            head; activate it per-call with forward(x, project=True).
        proj_dim: output dimension of the projection head.
    """
 
    def __init__(
        self,
        feat_dim: int = 256,
        widths: tuple = (32, 64, 128, 256, 256),
        in_channels: int = 1,
        dropout: float = 0.1,
        ssl_projection: bool = False,
        proj_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        w1 = widths[0]
 
        # Stem: stride 1 -- keep full resolution for thin strokes.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )
 
        # Five residual stages; stage 1 keeps resolution, stages 2-5 halve it.
        in_chs = (w1,) + tuple(widths[:-1])
        downsamples = (False, True, True, True, True)
        self.stages = nn.ModuleList()
        for in_ch, out_ch, down in zip(in_chs, widths, downsamples):
            stride = 2 if down else 1
            stage = nn.ModuleDict(
                {
                    "conv1": nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                       padding=1, bias=False),
                    "bn1": nn.BatchNorm2d(out_ch),
                    "conv2": nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                       padding=1, bias=False),
                    "bn2": nn.BatchNorm2d(out_ch),
                    # 1x1 projection on the skip path when shape changes,
                    # identity otherwise.
                    "skip": (
                        nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                                      bias=False),
                            nn.BatchNorm2d(out_ch),
                        )
                        if down or in_ch != out_ch
                        else nn.Identity()
                    ),
                }
            )
            self.stages.append(stage)
 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(widths[-1], feat_dim)
        self.feat_dim = feat_dim
 
        # Optional SimCLR-style projection head for SSL pre-training.
        self.projector = (
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, proj_dim),
            )
            if ssl_projection
            else None
        )
 
    def forward(self, x: torch.Tensor, project: bool = False) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            identity = stage["skip"](x)
            out = F.relu(stage["bn1"](stage["conv1"](x)), inplace=True)
            out = stage["bn2"](stage["conv2"](out))
            x = F.relu(out + identity, inplace=True)
 
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.head(x)
 
        if project:
            if self.projector is None:
                raise RuntimeError(
                    "Projection head not built; construct the model with "
                    "ssl_projection=True to use forward(x, project=True)."
                )
            x = F.normalize(self.projector(x), dim=1)
        return x
