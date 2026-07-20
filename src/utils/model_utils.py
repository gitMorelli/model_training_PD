import tarfile
import time
import io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import torch.nn as nn
import time
import webdataset as wds
import glob
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
from transformers import VisionEncoderDecoderModel, ViTModel, ViTForImageClassification
import torch.nn.functional as F

#custom models
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


#custom time series models

#custom classification heads
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super(CustomMLP, self).__init__()
        layers = []
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', None)
        batchnorm = kwargs.get('batchnorm', False)
        with_input_norm = kwargs.get('with_input_norm', None)
        if with_input_norm is None:
            pass
        elif with_input_norm=='batch_norm':
            layers.append(nn.BatchNorm1d(input_size))
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            if dropout is not None:
                if isinstance(dropout, float):
                    layers.append(nn.Dropout(dropout))
                else:
                    raise ValueError("Dropout should be a float value.")
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
class CustomTransformer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomTransformer, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_sizes[0], nhead=8),
            num_layers=len(hidden_sizes) - 1
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x
class Custom1DCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Custom1DCNN, self).__init__()
        layers = []
        in_channels = 1  # Assuming input is a single channel (e.g., grayscale)
        for hidden_size in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = hidden_size
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * (input_size // (2 ** len(hidden_sizes))), output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.unsqueeze(1))  # Add channel dimension
class CustomLogreg(nn.Module):
    def __init__(self, input_size, output_size,**kwargs):
        super(CustomLogreg, self).__init__()
        layers = []
        dropout = kwargs.get('dropout', None)
        batchnorm = kwargs.get('batchnorm', False)
        with_input_norm = kwargs.get('with_input_norm', None)
        if with_input_norm is None:
            pass
        elif with_input_norm=='batch_norm':
            layers.append(nn.BatchNorm1d(input_size))
        if dropout is not None:
            if isinstance(dropout, float):
                layers.append(nn.Dropout(dropout))
            else:
                raise ValueError("Dropout should be a float value.")
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#Pretrained CNN models
def get_resnet(name,mode, pretrained, **kwargs):
    def load_ln_checkpoint_for_resnet(full_model,checkpoint,custom_pre_trained_weights):
        if custom_pre_trained_weights:
            state_dict = checkpoint['state_dict'] 
            # Strip the "model." prefix added by LightningModule
            stripped = {k.removeprefix('model.'): v for k, v in state_dict.items()}
            keys_to_remove = [k for k in stripped if k.startswith('fc.')]
            print("Removing keys:", keys_to_remove)  # sanity check
            for k in keys_to_remove:
                stripped.pop(k)
            result = full_model.load_state_dict(stripped, strict=False)  # skips fc.weight / fc.bias 
            print("Missing keys:", result.missing_keys)    # in model but not in checkpoint
            print("Unexpected keys:", result.unexpected_keys)  # in checkpoint but not in model
        return full_model
    
    from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights
    custom_pre_trained_weights = kwargs.get('custom_pre_trained_weights', None)
    if custom_pre_trained_weights:
        checkpoint = torch.load(custom_pre_trained_weights)
        pretrained = False
    if name in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = resnet34(weights=weights)
        #full_model.load_state_dict(checkpoint['model_state_dict']) if custom_pre_trained_weights else None #torch
        full_model = load_ln_checkpoint_for_resnet(full_model,checkpoint, custom_pre_trained_weights)
        if name=='resnet34_layer1':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1
            )
        elif name=='resnet34_layer2':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2
            )
        elif name=='resnet34_layer3':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3
            )
            
        class WrappedResNet(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
                self.gap = torch.nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                x = self.layers(x)
                x = self.gap(x)  # [B, C, 1, 1]
                x = torch.flatten(x, 1)
                return x
        model = WrappedResNet(layers)
        return model
    elif name in ['resnet50','resnet18','resnet34','resnet101']:
        if name=='resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
        elif name=='resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
        elif name=='resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
        elif name=='resnet101':
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = resnet101(weights=weights)
        #model.load_state_dict(checkpoint['model_state_dict']) if custom_pre_trained_weights else None
        if custom_pre_trained_weights:
            model = load_ln_checkpoint_for_resnet(model,checkpoint, custom_pre_trained_weights)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', 'resnet18']")
    return model
def get_resnet_transforms(**kwargs):
    """
    Returns the transformation pipeline for ResNet.
    """
    mode=kwargs.get('mode',)
    if mode=='resize':
        transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform
def simple_resize_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

#Pretrained transformer models
def get_clip_vit(name, **kwargs):
    from transformers import CLIPModel
    normalization=True if name=="clip-vit-large-patch14" else False
    if name=="clip-vit-large-patch14-un":
        name="clip-vit-large-patch14"
    class WrappedModelInter(nn.Module):
      """
      Uses register_forward_hook to extract CLS from a specific
      encoder layer — works regardless of transformers version.
      """
      def __init__(self, model, layer_index: int = 12):
          super().__init__()
          self.vision = getattr(model, "vision_model", model)

          num_layers = self.vision.config.num_hidden_layers
          assert 1 <= layer_index <= num_layers, \
              f"layer_index must be in [1, {num_layers}]"

          self._captured = None

          # Hook fires after encoder.layers[layer_index - 1] completes
          # layer_index=12 → layers[11] → output after block 12
          target_layer = self.vision.encoder.layers[layer_index - 1]
          self._hook = target_layer.register_forward_hook(self._capture_hook)

      def _capture_hook(self, module, input, output):
          # CLIPEncoderLayer returns a tuple; index 0 is the hidden state
          hidden = output[0] if isinstance(output, tuple) else output
          self._captured = hidden

      def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
          self._captured = None
          # Just run the full vision model — the hook captures mid-way
          self.vision(pixel_values=pixel_values)
          assert self._captured is not None, "Hook did not fire"
          return self._captured[:, 0, :]  # CLS token

      def remove_hook(self):
          """Call this when done to avoid memory leaks.
          Memory leaks are a problem if you re-initialize the model multiple times in a notebook."""
          self._hook.remove()
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, type_of_output='cls',normalization=False):
            super().__init__()
            self.model = model
            self.type_of_output = type_of_output
            self.normalization = normalization

        def forward(self, x):
            image_features = self.model.get_image_features(x)
            image_features = image_features.last_hidden_state[:, 0, :]  # CLS token
            # Normalize the features (optional but common)
            if self.normalization:
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features
    class WrappedVisionModelExpl(torch.nn.Module):
        def __init__(self, vision_model, type_of_output='cls', normalization=False):
            super().__init__()
            self.vision_model = vision_model
            self.type_of_output = type_of_output  # Can be 'cls' or 'mean'
            self.normalization = normalization

        def forward(self, x):
            # Forward pass through the vision model
            outputs = self.vision_model(pixel_values=x, output_attentions=True)

            # Choose how to handle the output
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

            if self.type_of_output == 'cls':
                image_features = last_hidden[:, 0]  # CLS token
            elif self.type_of_output == 'mean':
                image_features = last_hidden.mean(dim=1)  # Mean pooling
            if self.normalization:
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            return image_features
    
    if 'inter' in name:
        name_temp=name.replace('-inter','')
    else:
        name_temp=name
    model = CLIPModel.from_pretrained(f'openai/{name_temp}')
    #model = WrappedHuggingfaceModel(model.vision_model) 
    #remove pixel_values argument; returns the output of the last layernorm (no pooling) 1,578,384
    
    if 'inter' in name:
        return WrappedModelInter(model, layer_index=12)
    else:
        return WrappedModel(model,'cls',normalization) #add an option for other ways of reading the output
        #return WrappedVisionModelExpl(model.vision_model, 'cls',normalization=normalization)
    #return WrappedVisionModelExpl(model.vision_model, type_of_output=kwargs.get('type_of_output', 'cls'),normalization=normalization)
def get_clip_vit_transforms(name, **kwargs):
    from transformers import CLIPImageProcessor
    if name == "clip-vit-large-patch14-un":
        name = "clip-vit-large-patch14"
    if 'inter' in name:
        name = name.replace('-inter','')
    processor = CLIPImageProcessor.from_pretrained(f"openai/{name}")
    return processor
def get_swin(name, mode, pretrained, **kwargs):
    if name == "swin_t":
        from torchvision.models import swin_t, Swin_T_Weights
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
    elif name == "swin_s":
        from torchvision.models import swin_s, Swin_S_Weights
        weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_s(weights=weights)
    elif name == "swin_b":
        from torchvision.models import swin_b, Swin_B_Weights
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_b(weights=weights)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2) 
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.head.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.head = mlp
    model.head = torch.nn.Identity()
    return model
def get_swin_transforms(name='swin_s',**kwargs):
    transform = transforms.Compose([
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

#Model loading
def get_model(name="resnet50", mode='classification head', pretrained=True,checkpoint_path=None, **kwargs):
    ''' 
    - name: the name of the model to download/load 
    -mode: 1) classification_head (modifies the last layers of the loaded model and appends an mlp classifier to the model)
    2) as is (loads the model as is, without any modifications)
    3) truncated (truncates the model to a certain number of layers) so that it returns an hidden representation    - pretrained: whether to load the pretrained weights or not
    - '''
    ### CNN MODELS ###
    if name.startswith('resnet'):
        model = get_resnet(name,mode, pretrained, **kwargs)
        transform = get_resnet_transforms(**kwargs)
    elif name.startswith('clip-vit'):
        model = get_clip_vit(name, **kwargs)
        transform = get_clip_vit_transforms(name, **kwargs) 
    elif name.startswith('swin'):
        model = get_swin(name, mode, pretrained, **kwargs)
        transform = get_swin_transforms(name, **kwargs)
    elif name == 'custom_cnn':
        model = CustomBinaryCNN() 
        input_size = kwargs.get('input_size', 224)
        transform = simple_resize_transform(input_size)
    elif name == 'FiveStageResidualStridedConvNet':
        model = FiveStageResidualStridedConvNet(**kwargs)
        input_size = kwargs.get('input_size', 224)
        transform = simple_resize_transform(input_size)
    else:
        raise ValueError(f"Model {name} is not supported.")
    #pretrained_modality = kwargs.get('custom_pretrained','original')
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, transform
def get_sklearn_model(name='logreg', **kwargs): 
    if name=='svm':
        from sklearn.svm import SVC
        C=kwargs.get('C',1.0)
        return SVC(kernel='rbf', C=C, gamma='scale', probability=True, random_state=42)
    elif name=='logreg':
        from sklearn.linear_model import LogisticRegression
        penalty=kwargs.get('penalty','l2')
        C=kwargs.get('C',1.0)
        solver=kwargs.get('solver','lbfgs')
        max_iter=kwargs.get('max_iter',5000)
        return LogisticRegression(max_iter=max_iter, random_state=42, penalty=penalty, C=C, solver=solver)
    elif name=='gbm':
        # Define the models
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, #100 is standard 
            learning_rate=0.1,  
            max_depth=3,  
            random_state=42
        )
    elif name=='lgbm':
        import lightgbm as lgb
        from lightgbm import early_stopping, log_evaluation
        return lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=20,
            min_child_samples=30,#Minimum number of data samples per leaf
            subsample=0.8, #Randomness in row 
            colsample_bytree=0.8, #and feature sampling respectively.
            reg_alpha=1.0, # L1 regularization
            reg_lambda=1.0, # L2 regularization
            random_state=42,
            n_jobs=-1,
            min_split_gain=0.01,  # Minimum gain to make a split
        )
    elif name=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    #rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    elif name=='rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,            # More trees = more stable
            max_depth=10,                # Limits tree depth (main regularizer)
            min_samples_split=10,        # Minimum samples to split a node
            min_samples_leaf=5,          # Minimum samples at a leaf node
            max_features='sqrt',         # Random feature selection at each split
            bootstrap=True,              # Use bootstrapped samples (default)
            oob_score=True,              # Out-of-bag error estimate
            random_state=42,
            n_jobs=-1
        )
    elif name=='mlp':
        from sklearn.neural_network import MLPClassifier
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', 256)
        return MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation='relu', solver='adam',
                            max_iter=200, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
    elif name=='dt':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, ccp_alpha=0.01, random_state=42)
def get_classification_head(name='MLPClassifier1',in_features=512,num_classes=2,**kwargs):
    dropout = kwargs.get('dropout', 0.5)
    activation = kwargs.get('activation', 'relu')
    n_neurons = kwargs.get('n_neurons', 64)
    with_input_norm = kwargs.get('with_input_norm', None)
    scale = kwargs.get('scale', 1.0)
    mean = kwargs.get('mean', 0.0)
    if name == 'MLPClassifier1': #1 hidden layer
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,
                         dropout=dropout, activation=activation, with_input_norm=with_input_norm,scale=scale, mean=mean)
    elif name == 'MLPClassifier2':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,
                         dropout=dropout, activation=activation,with_input_norm=with_input_norm,scale=scale, mean=mean)
    elif name == 'MLPClassifier3':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons,n_neurons])
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,dropout=dropout, 
                         activation=activation,with_input_norm=with_input_norm,scale=scale, mean=mean)
    if name == 'MLPClassifier1-BatchNorm': #1 hidden layer
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes, activation='relu',batchnorm=True)
    elif name == 'MLPClassifier2-BatchNorm':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons])
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes, dropout=0.2 ,activation='relu',batchnorm=True)
    elif name == 'TransformerClassifier':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons])
        return CustomTransformer(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
    elif name == '1DCNNClassifier':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons])
        return Custom1DCNN(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
    elif name in ['logreg','linear']:
        return CustomLogreg(input_size=in_features, output_size=num_classes)
    elif name=='regularized_linear':
        return CustomLogreg(input_size=in_features, output_size=num_classes, dropout=dropout, batchnorm=True, with_input_norm='batch_norm')
    else:
        raise ValueError(f"Classification head {name} is not supported. Choose from ['MLPClassifier1', 'MLPClassifier2', 'TransformerClassifier']")
def unfreeze_layers(model,layer_names=['all']):
    if len(layer_names) == 0:
        layer_names = ['frozen']
    for name, param in model.named_parameters():
        if layer_names[0]=='all' or any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
def load_backbone_from_lightning_ckpt(backbone, ckpt_path):
    # 1. Load the checkpoint file
    # Using map_location='cpu' prevents out-of-memory errors if loading a GPU model on CPU
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # 2. Extract the state dictionary
    full_state_dict = checkpoint['state_dict']
    
    # 3. Create a new dictionary for just the backbone weights
    backbone_state_dict = {}
    
    for key, weight in full_state_dict.items():
        # Look for keys that belong to your backbone
        if 'vision_model' in key:
            # PyTorch Lightning usually adds prefixes (e.g., "model.vision_model.layer1...").
            # Your standalone backbone just expects "layer1...".
            # We split by 'vision_model.' and take the right side to strip all prefixes.
            clean_key = key.split('vision_model.')[-1]
            
            backbone_state_dict[clean_key] = weight

    # 4. Load the filtered weights into your backbone
    # strict=True ensures that all keys match perfectly without any missing or extra weights.
    backbone.load_state_dict(backbone_state_dict, strict=True)
    
    print(f"Successfully loaded {len(backbone_state_dict)} tensors into the backbone.")
    return backbone

#Multiple instance learning modules
class GatedAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
    def forward(self, x):                                   # (N, k, F)
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        alpha = torch.softmax(a, dim=1)
        return (alpha * x).sum(1), alpha.squeeze(-1)

#Wrappers
class JoinedModels(nn.Module):
    def __init__(self, vision_model, classifier):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier

    def forward(self, x):
        features = self.vision_model(x)  # image -> features
        logits = self.classifier(features)  # features -> prediction
        return logits
class TiledJoinedModels(nn.Module):

    def __init__(self, vision_model, classifier):
        # Fixed the mismatch: super() should match the current class name
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier

    def forward(self, x):
        # x shape: (B, n, C, H, W)
        B, n, C, H, W = x.size()

        # 1. Collapse Batch and Tile dimensions into a single batch
        # New shape: (B * n, C, H, W)
        x_flattened = x.view(B * n, C, H, W)

        # 2. Forward pass through vision model in parallel (1 call instead of n)
        # Expected output shape per image: (B * n, feature_dim)
        features_flattened = self.vision_model(x_flattened)

        # 3. Reshape back to separate Batch and Tiles, then flatten tiles into features
        # Shape transition: (B * n, feature_dim) -> (B, n * feature_dim)
        feature_dim = features_flattened.size(-1)
        combined_feats = features_flattened.view(B, n * feature_dim)

        # 4. Final classification
        return self.classifier(combined_feats)
class ConcatenateViews(nn.Module):
    def __init__(self, vision_model):
        # Fixed the mismatch: super() should match the current class name
        super().__init__()
        self.vision_model = vision_model

    def forward(self, x):
        # x shape: (B, n, C, H, W)
        B, n, C, H, W = x.size()

        # 1. Collapse Batch and Tile dimensions into a single batch
        # New shape: (B * n, C, H, W)
        x_flattened = x.view(B * n, C, H, W)

        # 2. Forward pass through vision model in parallel (1 call instead of n)
        # Expected output shape per image: (B * n, feature_dim)
        features_flattened = self.vision_model(x_flattened)

        # 3. Reshape back to separate Batch and Tiles, then flatten tiles into features
        # Shape transition: (B * n, feature_dim) -> (B, n * feature_dim)
        feature_dim = features_flattened.size(-1)
        features = features_flattened.view(B, n, feature_dim)  # ← Changed this line

        
        return features


#Time-sequence models
class SequenceClassifierHead(nn.Module):
    """Everything after the CNN: view aggregation + slot transformer + classification.
    All params land under `classifier.*` in the parent's named_parameters()."""
    def __init__(self, feat_dim, n_slots, n_classes,
                 d_model=512, n_heads=8, n_layers=4, ff_mult=4,
                 view_agg='attention', dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()

        self.view_agg  = view_agg
        self.view_pool = GatedAttentionPool(d_model) if view_agg == 'attention' else None

        self.n_slots  = n_slots
        self.slot_pos = nn.Embedding(n_slots, d_model)
        self.cls      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, feats, N, k, seq_ids, slot_ids, lengths, return_view_attn=False):
        # N total number of frames (can come from different subjects)
        # feats: (N*k, feat_dim) straight from the CNN
        x = self.proj(feats).view(N, k, -1)                 # (N, k, d_model)

        if self.view_pool is not None:
            q_repr, view_attn = self.view_pool(x)           # (N, d_model)
        else:
            q_repr, view_attn = x.mean(1), None

        B   = lengths.size(0)
        D   = q_repr.size(-1)
        dev = q_repr.device

        #these lines, for each subject, select the available timesteps and put them in a buffer of size (B, n_slots, D) with padding for missing slots
        #-> they manage to transfmr a sum(T_i) x D tensor into a B x n_slots x D tensor, where T_i is the number of available slots for subject i
        buffer = q_repr.new_zeros(B, self.n_slots, D)
        buffer[seq_ids, slot_ids] = q_repr
        pad_mask = torch.ones(B, self.n_slots, dtype=torch.bool, device=dev)
        pad_mask[seq_ids, slot_ids] = False

        pos    = self.slot_pos(torch.arange(self.n_slots, device=dev))
        buffer = buffer + pos.unsqueeze(0)

        cls  = self.cls.expand(B, -1, -1)
        seq  = torch.cat([cls, buffer], dim=1)              # (B, 1+Q, D)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev),
                          pad_mask], dim=1)

        out    = self.transformer(seq, src_key_padding_mask=mask)
        logits = self.head(self.norm(out[:, 0]))
        return (logits, view_attn) if return_view_attn else logits
class SequenceQuestionnaireModel(nn.Module):
    def __init__(self, vision_model, feat_dim, n_slots, n_classes, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model                               # -> cnn.*
        self.classifier = SequenceClassifierHead(           # -> classifier.*
            feat_dim, n_slots, n_classes, **head_kwargs)

    def forward(self, frames, seq_ids, slot_ids, lengths, return_view_attn=False):
        N, k  = frames.shape[:2]
        feats = self.vision_model(frames.flatten(0, 1))             # (N*k, feat_dim)  <- only heavy step
        return self.classifier(feats, N, k, seq_ids, slot_ids, lengths, return_view_attn)
#unordered models
class SetClassifierHead(nn.Module):
    """Order-invariant head over a variable number of timesteps per subject.

        logits = rho( pool_{t in available} phi(x_t) )

    Pooling is done by segment-reduction over seq_ids, so every timestep
    handed to the head is used: there is no capacity cap, no padding buffer,
    and no slot_ids. Subjects may have any count >= 1, and counts may differ
    within a batch and between train and inference.

    Pooled features fed to rho:
      * mean of phi(x_t) over the subject's timesteps
      * std across timesteps (optional); zero when the subject has one timestep
      * normalized count (optional)
    """

    def __init__(self, feat_dim, n_classes,
                 d_model=512, view_agg='attention', dropout=0.1,
                 ff_mult=2, use_spread=True, use_count_feature=True,
                 count_norm=8.0):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()
        self.view_agg = view_agg
        self.view_pool = GatedAttentionPool(d_model) if view_agg == 'attention' else None

        # shared per-timestep encoder (phi)
        self.phi = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.use_spread = use_spread
        self.use_count_feature = use_count_feature
        self.count_norm = float(count_norm)   # scaling only, NOT a capacity limit

        rho_in = d_model \
            + (d_model if use_spread else 0) \
            + (1 if use_count_feature else 0)

        # set-level classifier (rho)
        self.rho = nn.Sequential(
            nn.LayerNorm(rho_in),
            nn.Linear(rho_in, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, n_classes),
        )

    def forward(self, feats, N, k, seq_ids, lengths, return_view_attn=False):
        # feats: (N*k, feat_dim) straight from the CNN
        # N = sum(T_i) = total timesteps in the batch
        # seq_ids: (N,) which subject each timestep belongs to
        x = self.proj(feats).view(N, k, -1)                 # (N, k, d_model)
        if self.view_pool is not None:
            q_repr, view_attn = self.view_pool(x)           # (N, d_model)
        else:
            q_repr, view_attn = x.mean(1), None

        B   = lengths.size(0)
        D   = q_repr.size(-1)
        dev = q_repr.device
        seq = seq_ids.long()

        assert int(seq.max()) < B, \
            f"seq_id {int(seq.max())} out of range for batch of {B} subjects"

        h = self.phi(q_repr)                                # (N, D)

        # segment-reduce over subjects: every timestep in the batch contributes
        n = torch.zeros(B, 1, device=dev, dtype=h.dtype).index_add_(
            0, seq, torch.ones(seq.numel(), 1, device=dev, dtype=h.dtype))
        assert (n.squeeze(1) > 0).all(), "a subject in the batch has no timesteps"

        s = torch.zeros(B, D, device=dev, dtype=h.dtype).index_add_(0, seq, h)
        mean = s / n

        parts = [mean]
        if self.use_spread:
            # E[h^2] - E[h]^2, in one pass; clamp guards fp cancellation
            sq = torch.zeros(B, D, device=dev, dtype=h.dtype).index_add_(0, seq, h * h)
            var = (sq / n - mean ** 2).clamp(min=0)
            parts.append(torch.sqrt(var + 1e-8))            # ~zero when n == 1
        if self.use_count_feature:
            parts.append(((n - 1.0) / (self.count_norm - 1.0)).clamp(0, 1))

        logits = self.rho(torch.cat(parts, dim=-1))
        return (logits, view_attn) if return_view_attn else logits
class SetQuestionnaireModel(nn.Module):
    def __init__(self, vision_model, feat_dim, n_classes, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model                    # -> cnn.*
        self.classifier = SetClassifierHead(                # -> classifier.*
            feat_dim, n_classes, **head_kwargs)

    def forward(self, frames, seq_ids, slot_ids, lengths, return_view_attn=False):
        N, k  = frames.shape[:2]
        feats = self.vision_model(frames.flatten(0, 1))     # (N*k, feat_dim)
        return self.classifier(feats, N, k, seq_ids, lengths, return_view_attn)

#others
def test_output(size, model,channels=3):
    dummy_input = torch.rand(1, channels, size, size)
    dummy_input.shape
    '''if huggingface:
        # the transform is actually an huggingface processor in this case
        inputs = transform(images=dummy_input, return_tensors="pt")
        # Remove batch dimension from inputs
        patch = inputs['pixel_values'].squeeze()
    else:
        patch = transform(dummy_input)'''
    #print(model)
    with torch.no_grad():
        output = model(dummy_input)
    return output

 #

