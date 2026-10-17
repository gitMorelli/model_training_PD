from torch import nn
import torch

# get lora enabled models #
def _timm_eval_transform(model, mode, in_chans):
    from timm.data import resolve_model_data_config, create_transform

    cfg = resolve_model_data_config(model)
    if in_chans == 1:
        # average the 3-channel stats to match the summed RGB stem/patch weights
        cfg['mean'] = (sum(cfg['mean']) / len(cfg['mean']),)
        cfg['std'] = (sum(cfg['std']) / len(cfg['std']),)
        cfg['input_size'] = (1, *cfg['input_size'][1:])
    if mode == 'resize':
        cfg.update(crop_pct=1.0, crop_mode='squash')   # square resize, no center crop
    return create_transform(**cfg, is_training=False)
def get_convnext_lora(name, mode, pretrained, **kwargs):
    """
    timm ConvNeXt backbone returning pooled, normalized features [B, out_dim].

      - num_classes=0  -> head.fc is Identity; forward() = global pool -> LayerNorm -> flatten
      - in_chans=1     -> timm sums the pretrained RGB stem weights (same as the old grayscale_version)
      - _layerK suffix -> keep the first K stages, normalized with the pretrained LayerNorm2d
                          at the front of stage K+1's downsample (trained on exactly this output)
    """
    import timm

    # aliases so existing configs keep working; any timm model name also passes through
    aliases = {
        'convnext_tiny':  'convnext_tiny.fb_in1k',
        'convnext_small': 'convnext_small.fb_in1k',
        'convnext_base':  'convnext_base.fb_in1k',
        'convnext_large': 'convnext_large.fb_in1k',
    }

    grayscale = kwargs.get('grayscale', False)
    use_final_norm = kwargs.get('final_norm', True)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)

    base_name, _, suffix = name.partition('_layer')
    timm_name = aliases.get(base_name, base_name)

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=0,
        in_chans=1 if grayscale else 3,
        drop_path_rate=drop_path_rate,
    )

    n_stages = len(model.stages)
    k = int(suffix) if suffix else n_stages
    if not 1 <= k <= n_stages:
        raise ValueError(f"Unsupported stage suffix in {name}. Use _layer1 ... _layer{n_stages}.")

    if k < n_stages:
        # stages[k].downsample = Sequential(LayerNorm2d(C_k), Conv2d(C_k, C_k+1, 2, 2))
        norm = model.stages[k].downsample[0]
        model.stages = nn.Sequential(*list(model.stages[:k]))
        model.head.norm = norm
        model.num_features = norm.normalized_shape[0]

    if not use_final_norm:
        model.head.norm = nn.Identity()

    model.out_dim = model.num_features
    # pointwise MLP linears inside every block (where most of ConvNeXt's parameters are)
    model.lora_target_modules = r"stages\.\d+\.blocks\.\d+\.mlp\.(fc1|fc2)"
    return model
def get_convnext_lora_transforms(**kwargs):
    model = kwargs.get('model', None)
    if model is None:
        raise ValueError("pass model=<the model from get_convnext> so the config matches the checkpoint")
    return _timm_eval_transform(model, kwargs.get('mode'), model.stem[0].in_channels)

def get_vit_lora(name, mode, pretrained, **kwargs):
    """
    timm ViT backbone returning pooled features [B, out_dim].

    No wrapper class and no grayscale surgery needed:
      - num_classes=0  -> the head is dropped, forward() returns pooled features
      - in_chans=1     -> timm rebuilds the patch conv and sums the pretrained
                          RGB weights across channels (exactly our old hand-written
                          grayscale_version, but done inside load_pretrained)
      - img_size=N     -> position embeddings are interpolated automatically
    """
    import timm

    # aliases so existing configs keep working; any timm model name also passes through
    aliases = {
        'vit_b_16': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'vit_b_32': 'vit_base_patch32_224.augreg_in21k_ft_in1k',
        'vit_s_16': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'vit_l_16': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
    }

    grayscale = kwargs.get('grayscale', False)
    pool = kwargs.get('pool', 'token')          # 'token' | 'avg' | 'avgmax' | ''
    image_size = kwargs.get('image_size', None)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)

    base_name, _, suffix = name.partition('_layer')
    timm_name = aliases.get(base_name, base_name)

    create_kwargs = dict(
        pretrained=pretrained,
        num_classes=0,
        in_chans=1 if grayscale else 3,
        global_pool=pool,
        drop_path_rate=drop_path_rate,
    )
    if image_size is not None:
        create_kwargs['img_size'] = image_size

    model = timm.create_model(timm_name, **create_kwargs)

    if suffix:
        k = int(suffix)
        if k not in (1, 2, 3):
            raise ValueError(f"Unsupported stage suffix in {name}. Use _layer1, _layer2 or _layer3.")
        n_blocks = len(model.blocks)
        cut = max(1, k * n_blocks // 4)
        model.blocks = nn.Sequential(*list(model.blocks[:cut]))
        # model.norm was calibrated on the full depth; ViT blocks are pre-norm, so a
        # truncated trunk returns an unnormalized residual stream -> fresh LayerNorm
        model.norm = nn.LayerNorm(model.embed_dim, eps=model.norm.eps)

    model.out_dim = model.num_features
    model.lora_target_modules = r"blocks\.\d+\.attn\.(qkv|proj)"

    return model
def get_vit_lora_transforms(**kwargs):
    model = kwargs.get('model', None)
    if model is None:
        raise ValueError("pass model=<the model from get_vit> so the config matches the checkpoint")
    return _timm_eval_transform(model, kwargs.get('mode'), model.patch_embed.proj.in_channels)
