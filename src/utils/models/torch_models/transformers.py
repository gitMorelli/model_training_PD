from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

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
class WrappedSwin(nn.Module):
    """
    Swin backbone without the Linear head.

    forward: features -> LayerNorm -> permute -> global average pool -> flatten -> [B, out_dim]

    NOTE the ordering. torchvision's Swin runs `features` in channels-LAST layout
    ([B, H, W, C]) and applies `self.norm` BEFORE pooling:
        features -> norm -> permute(0,3,1,2) -> avgpool -> flatten -> head
    LayerNorm over C does not commute with the spatial mean, so norm-then-pool is
    not interchangeable with pool-then-norm (unlike ConvNeXt, where the norm sits
    after the pool and both are equivalent on a 1x1 map).

    Submodule names, for optimizer grouping:
        features.0        patch embedding (4x4 s4 conv + permute + LayerNorm)
        features.1        stage 1
        features.2/3      PatchMerging + stage 2
        features.4/5      PatchMerging + stage 3
        features.6/7      PatchMerging + stage 4
        final_norm        LayerNorm(C)
    """

    def __init__(self, features, final_norm, out_dim):
        super().__init__()
        self.features = features
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)         # [B, H, W, C]  -- channels last
        x = self.final_norm(x)       # LayerNorm over C
        x = x.permute(0, 3, 1, 2)    # [B, C, H, W]
        x = self.gap(x)              # [B, C, 1, 1]
        x = self.flatten(x)          # [B, C]
        return x
def get_swin(name, mode, pretrained, **kwargs):
    def grayscale_version(model):
        # patch embed: features[0] is Sequential -> [0] is Conv2d(3, embed_dim, 4, 4)
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
        swin_t, Swin_T_Weights, swin_s, Swin_S_Weights, swin_b, Swin_B_Weights,
        swin_v2_t, Swin_V2_T_Weights, swin_v2_s, Swin_V2_S_Weights, swin_v2_b, Swin_V2_B_Weights,
    )

    builders = {
        'swin_t':    (swin_t,    Swin_T_Weights.IMAGENET1K_V1),
        'swin_s':    (swin_s,    Swin_S_Weights.IMAGENET1K_V1),
        'swin_b':    (swin_b,    Swin_B_Weights.IMAGENET1K_V1),
        'swin_v2_t': (swin_v2_t, Swin_V2_T_Weights.IMAGENET1K_V1),
        'swin_v2_s': (swin_v2_s, Swin_V2_S_Weights.IMAGENET1K_V1),
        'swin_v2_b': (swin_v2_b, Swin_V2_B_Weights.IMAGENET1K_V1),
    }

    # features = [patch_embed, stage1, merge, stage2, merge, stage3, merge, stage4]
    # same 8-entry layout as ConvNeXt, so cutting after stage k means features[:cut]
    cuts = {'layer1': 2, 'layer2': 4, 'layer3': 6, 'layer4': 8}
    # Output dims: t/s -> 96/192/384/768 per stage;  b -> 128/256/512/1024. Same for v2.

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

    if cut == 8:
        # the pretrained final LayerNorm(C)
        norm = model.norm
        out_dim = norm.normalized_shape[0]
    else:
        # PatchMerging's own norm is LayerNorm(4*dim) over the 2x2-concatenated
        # features, so it cannot be reused here the way ConvNeXt's can. Swin blocks
        # are pre-norm, so a truncated trunk returns an unnormalized residual stream
        # -> a fresh LayerNorm is worth having, it just starts at weight=1, bias=0.
        out_dim = model.features[cut].reduction.in_features // 4
        norm = nn.LayerNorm(out_dim)

    return WrappedSwin(features, norm if use_final_norm else None, out_dim)
def get_swin_transforms(**kwargs):
    """
    Returns the transformation pipeline for Swin.
    """
    mode = kwargs.get('mode',)
    name = kwargs.get('name', 'swin_t')
    base_name = name.partition('_layer')[0]

    # crop / resize used by the torchvision ImageNet1K weights (all bicubic)
    meta = {
        'swin_t':    (224, 232),
        'swin_s':    (224, 246),
        'swin_b':    (224, 238),
        'swin_v2_t': (256, 260),
        'swin_v2_s': (256, 260),
        'swin_v2_b': (256, 272),
    }
    crop_size, resize_size = meta.get(base_name, (224, 232))
    crop_size = kwargs.get('image_size', crop_size)
    resize_size = kwargs.get('resize_size', resize_size if 'image_size' not in kwargs else crop_size)

    interpolation = transforms.InterpolationMode.BICUBIC

    gray_scale = kwargs.get('grayscale', False)
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
