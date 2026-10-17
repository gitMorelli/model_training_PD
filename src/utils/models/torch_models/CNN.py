from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

#Pretrained CNN models
class WrappedConvNeXt(nn.Module):
    """
    ConvNeXt backbone without the Linear head.
 
    forward: features -> global average pool -> LayerNorm2d -> flatten  ->  [B, out_dim]
 
    Submodule names, for optimizer grouping:
        features.0        stem
        features.1        stage 1
        features.2/3      downsample + stage 2
        features.4/5      downsample + stage 3
        features.6/7      downsample + stage 4
        final_norm        LayerNorm2d (pretrained, C affine params)
    """
 
    def __init__(self, features, final_norm, out_dim):
        super().__init__()
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.flatten = nn.Flatten(1)
        self.out_dim = out_dim
 
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)            # [B, C, 1, 1]
        x = self.final_norm(x)     # LayerNorm2d normalizes over C, so this is LayerNorm(C) here
        x = self.flatten(x)        # [B, C]
        return x
def get_convnext(name, mode, pretrained, **kwargs):
    def grayscale_version(model):
        # ConvNeXt stem: features[0] is a Conv2dNormActivation -> [0] is the Conv2d(3, C, 4, 4)
        old = model.features[0][0]
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None)
 
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))  # [C,3,4,4] -> [C,1,4,4]
            if old.bias is not None:
                new.bias.copy_(old.bias)
 
        model.features[0][0] = new
        return model
 
    from torchvision.models import (
        convnext_tiny, ConvNeXt_Tiny_Weights,
        convnext_small, ConvNeXt_Small_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )
 
    builders = {
        'convnext_tiny':  (convnext_tiny,  ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        'convnext_small': (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
        'convnext_base':  (convnext_base,  ConvNeXt_Base_Weights.IMAGENET1K_V1),
        'convnext_large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
    }
 
    # features = [stem, stage1, down2, stage2, down3, stage3, down4, stage4]
    # so cutting after stage k means features[:cut]
    cuts = {'layer1': 2, 'layer2': 4, 'layer3': 6, 'layer4': 8}
    # Output dims for tiny/small are 96/192/384/768 per stage; base 128/256/512/1024; large 192/384/768/1536.
 
    grayscale = kwargs.get('grayscale', False)
    use_final_norm = kwargs.get('final_norm', True)
 
    base_name, _, suffix = name.partition('_layer')
    if base_name not in builders:
        raise ValueError(
            f"Model {name} is not supported. Choose from "
            f"{list(builders)} optionally suffixed with _layer1/_layer2/_layer3."
        )
 
    builder, default_weights = builders[base_name]
    weights = default_weights if pretrained else None
    model = builder(weights=weights)
 
    if grayscale:
        model = grayscale_version(model)
 
    key = 'layer' + suffix if suffix else 'layer4'
    if key not in cuts:
        raise ValueError(f"Unsupported stage suffix in {name}. Use _layer1, _layer2 or _layer3.")
    cut = cuts[key]
 
    features = nn.Sequential(*list(model.features[:cut]))
 
    # Pick up a pretrained LayerNorm2d matched to the cut point:
    #   - full model  -> the norm at the front of the classifier
    #   - truncated   -> the norm at the front of the *next* downsample block,
    #                    which was trained on exactly this stage's output
    norm = model.classifier[0] if cut == 8 else model.features[cut][0]
    out_dim = norm.normalized_shape[0]
 
    return WrappedConvNeXt(features, norm if use_final_norm else None, out_dim)
def get_convnext_transforms(**kwargs):
    """
    Returns the transformation pipeline for ConvNeXt.
    """
    mode = kwargs.get('mode',)
    name = kwargs.get('name', 'convnext_tiny')
 
    # resize sizes used by the torchvision ImageNet1K weights
    resize_sizes = {
        'convnext_tiny': 236,
        'convnext_small': 230,
        'convnext_base': 232,
        'convnext_large': 232,
    }
    resize_size = resize_sizes.get(name.partition('_layer')[0], 232)

    gray_scale = kwargs.get('grayscale', False)
    if gray_scale:
        mean=[0.459]
        std=[0.226] 
        #luminance weighted sum
    else:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
 
    if mode == 'resize':
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transform
def get_resnet(name,mode, pretrained, **kwargs):
    def grayscale_version(model):
        old = model.conv1
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None)

        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))   # [64,3,7,7] -> [64,1,7,7]
            if old.bias is not None:
                new.bias.copy_(old.bias)

        model.conv1 = new
        return model
    def reduce_stride(model, modality='stride_red'):
        if modality=='stride_red':
            # Reduce stride of the first conv layer from 2 to 1
            model.conv1.stride = (1, 1)
            # Remove the maxpool layer
            model.maxpool = nn.Identity()
        elif modality=='conv3x3':
            old = model.conv1.weight.data  # (64, 3, 7, 7)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Initialise the new 3x3 kernel from the centre of the pretrained 7x7 kernel
            model.conv1.weight.data.copy_(old[:, :, 2:5, 2:5])
            model.maxpool = nn.Identity()
        return model
    from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights
    
    grayscale = kwargs.get('grayscale', False)
    if name in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
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
        
        model.fc = torch.nn.Identity()
        if grayscale:
            model = grayscale_version(model)
    elif name in ['resnet18_32']:
        if name=='resnet18_32':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
        
        model.fc = torch.nn.Identity()
        
        model = reduce_stride(model, modality='conv3x3') #use stride=1 instead of 2 

        if grayscale:
            model = grayscale_version(model)
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

class WrappedEfficientNetV2(nn.Module):
    """
    EfficientNetV2 backbone without the Linear head.

    forward: features -> global average pool -> flatten  ->  [B, out_dim]

    There is no trailing norm to keep here: every stage already ends in a
    BatchNorm (the MBConv project conv), and the full model ends in a
    Conv2dNormActivation, so the pooled vector is already normalized.
    The pretrained head is just Dropout + Linear, both of which we drop.

    Submodule names, for optimizer grouping:
        features.0            stem (3x3 s2 conv + BN + SiLU)
        features.1..N         stages  (N = 6 for _s, 7 for _m and _l)
        features.N+1          final 1x1 conv to 1280 channels (full model only)
    """

    def __init__(self, features, out_dim):
        super().__init__()
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)        # [B, C, 1, 1]
        x = self.flatten(x)    # [B, C]
        return x
def get_efficientnet_v2(name, mode, pretrained, **kwargs):
    def grayscale_version(model):
        # stem: features[0] is a Conv2dNormActivation -> [0] is Conv2d(3, C, 3, s=2, p=1, bias=False)
        old = model.features[0][0]
        new = nn.Conv2d(1, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=old.bias is not None)

        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))  # [C,3,3,3] -> [C,1,3,3]
            if old.bias is not None:
                new.bias.copy_(old.bias)

        model.features[0][0] = new
        return model

    from torchvision.models import (
        efficientnet_v2_s, EfficientNet_V2_S_Weights,
        efficientnet_v2_m, EfficientNet_V2_M_Weights,
        efficientnet_v2_l, EfficientNet_V2_L_Weights,
    )

    builders = {
        'efficientnet_v2_s': (efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        'efficientnet_v2_m': (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
        'efficientnet_v2_l': (efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
    }

    # features = [stem, stage1, ..., stageN, final_1x1_conv]
    # so cutting after stage k means features[:k+1]. N is 6 for _s and 7 for _m / _l,
    # which means '_layer4' is NOT the same depth across variants.

    grayscale = kwargs.get('grayscale', False)

    base_name, _, suffix = name.partition('_layer')
    if base_name not in builders:
        raise ValueError(
            f"Model {name} is not supported. Choose from "
            f"{list(builders)} optionally suffixed with _layer1.._layerN."
        )

    builder, default_weights = builders[base_name]
    weights = default_weights if pretrained else None
    model = builder(weights=weights)

    if grayscale:
        model = grayscale_version(model)

    n_stages = len(model.features) - 2  # minus stem and final conv

    if suffix:
        k = int(suffix)
        if not 1 <= k <= n_stages:
            raise ValueError(
                f"Unsupported stage suffix in {name}: {base_name} has {n_stages} stages, "
                f"use _layer1.._layer{n_stages}."
            )
        features = nn.Sequential(*list(model.features[:k + 1]))
    else:
        features = model.features  # includes the final 1x1 conv -> 1280 channels

    # channel count of the last conv in the (possibly truncated) trunk;
    # for an MBConv/FusedMBConv block that is the project conv, which is what we want
    out_dim = [m.out_channels for m in features.modules() if isinstance(m, nn.Conv2d)][-1]

    return WrappedEfficientNetV2(features, out_dim)
def get_efficientnet_v2_transforms(**kwargs):
    """
    Returns the transformation pipeline for EfficientNetV2.
    """
    mode = kwargs.get('mode',)
    name = kwargs.get('name', 'efficientnet_v2_s')
    base_name = name.partition('_layer')[0]

    # crop / resize / interpolation used by the torchvision ImageNet1K weights
    meta = {
        'efficientnet_v2_s': (384, 384, transforms.InterpolationMode.BILINEAR, 'imagenet'),
        'efficientnet_v2_m': (480, 480, transforms.InterpolationMode.BILINEAR, 'imagenet'),
        'efficientnet_v2_l': (480, 480, transforms.InterpolationMode.BICUBIC,  'half'),
    }
    crop_size, resize_size, interpolation, norm_kind = meta.get(
        base_name, (384, 384, transforms.InterpolationMode.BILINEAR, 'imagenet')
    )

    # let the caller train at a cheaper resolution than the eval default
    crop_size = kwargs.get('image_size', crop_size)
    resize_size = kwargs.get('resize_size', resize_size if 'image_size' not in kwargs else crop_size)

    gray_scale = kwargs.get('grayscale', False)
    if norm_kind == 'half':
        # the _l weights were ported with mean=std=0.5, NOT the ImageNet statistics
        mean = [0.5] if gray_scale else [0.5, 0.5, 0.5]
        std = [0.5] if gray_scale else [0.5, 0.5, 0.5]
    else:
        if gray_scale:
            mean = [0.459]   # luminance weighted sum
            std = [0.226]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

    if mode == 'resize':
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transform
