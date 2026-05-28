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
    def __init__(self, input_size, output_size):
        super(CustomLogreg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

#Pretrained CNN models
def get_resnet(name,mode, pretrained, **kwargs):
    from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights
    if name=='resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
    elif name=='resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif name=='resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif name=='resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet101(weights=weights)
    elif name in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = resnet34(weights=weights)
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
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', 'resnet18']")
    if mode=='classification head':
        num_classes=kwargs.get('num_classes', 2)
        hidden_sizes=kwargs.get('hidden_sizes', [128])
        in_features = model.fc.in_features
        mlp=CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.fc = mlp
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if name not in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
                model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
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

#Pretrained transformer models
def get_clip_vit(name, mode, **kwargs):
    from transformers import CLIPModel
    normalization=True if name=="clip-vit-large-patch14" else False
    if name=="clip-vit-large-patch14-un":
        name="clip-vit-large-patch14"
    class WrappedModelInter(nn.Module):
        """
        Works with either:
        - transformers.CLIPVisionModel
        - transformers.CLIPModel  (uses .vision_model internally)
        Returns [CLS] from the specified vision block.
        """
        def __init__(self, model, layer_index: int = 12):
            super().__init__()
            # If it's a full CLIPModel, grab the vision tower
            self.vision = getattr(model, "vision_model", model)
            num_layers = self.vision.config.num_hidden_layers
            assert 1 <= layer_index <= num_layers, f"layer_index must be in [1, {num_layers}]"
            self.layer_index = layer_index

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Run ONLY the vision encoder; request hidden states
            out = self.vision(pixel_values=pixel_values, output_hidden_states=True)
            hs = out.hidden_states[self.layer_index]   # [B, seq_len, hidden_dim]
            cls = hs[:, 0, :]                          # [CLS]
            return cls
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, type_of_output='cls',normalization=False):
            super().__init__()
            self.model = model
            self.type_of_output = type_of_output
            self.normalization = normalization

        def forward(self, x):
            image_features = self.model.get_image_features(x)
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
    
    if mode=='classification head':
        num_classes=kwargs.get('num_classes', 2)
        hidden_sizes=kwargs.get('hidden_sizes', [128])
        how_to_read=kwargs.get('how_to_read', 'cls')
        if how_to_read=='cls':
            in_features = 384
        else:
            print('still no support for pooling or other averaging techniques')
        mlp=CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
    elif mode=='as is':
        return model
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if 'inter' in name:
                return WrappedModelInter(model, layer_index=12)
            else:
                return WrappedModel(model,'cls',normalization) #add an option for other ways of reading the output
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    elif mode=='exp':
        return WrappedVisionModelExpl(model.vision_model, type_of_output=kwargs.get('type_of_output', 'cls'),normalization=normalization)
def get_clip_vit_transforms(name, **kwargs):
    from transformers import CLIPImageProcessor
    if name == "clip-vit-large-patch14-un":
        name = "clip-vit-large-patch14"
    if 'inter' in name:
        name = name.replace('-inter','')
    processor = CLIPImageProcessor.from_pretrained(f"openai/{name}")
    return processor

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
        model = get_clip_vit(name, mode, pretrained, **kwargs)
        transform = get_clip_vit_transforms(name, **kwargs)  
    else:
        raise ValueError(f"Model {name} is not supported.")
    #pretrained_modality = kwargs.get('custom_pretrained','original')
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, transform
def get_classification_head(name='MLPClassifier1',in_features=512,num_classes=2,**kwargs):
    dropout = kwargs.get('dropout', None)
    activation = kwargs.get('activation', 'relu')
    n_neurons = kwargs.get('n_neurons', 128)
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
    elif name == 'logreg':
        return CustomLogreg(input_size=in_features, output_size=num_classes)
    else:
        raise ValueError(f"Classification head {name} is not supported. Choose from ['MLPClassifier1', 'MLPClassifier2', 'TransformerClassifier']")

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

#others
def test_output(size, model):
    dummy_input = torch.rand(1, 3, size, size)
    dummy_input.shape
    '''if huggingface:
        # the transform is actually an huggingface processor in this case
        inputs = transform(images=dummy_input, return_tensors="pt")
        # Remove batch dimension from inputs
        patch = inputs['pixel_values'].squeeze()
    else:
        patch = transform(dummy_input)'''
    with torch.no_grad():
        output = model(dummy_input)
    return output