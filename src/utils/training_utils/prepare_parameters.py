import os
from src.utils.models.pretrained_models.models import _process_name, mapping_conv_encoder, SOURCE_PATH, _load_preprocessing_params
from src.utils.data_loading import get_representation_file_path
from src.utils.data_loading import read_rep_dim

#optimization groups utils
def _vit_depth(base_name):
    # full (untruncated) depth of a timm ViT/DeiT from its name; covers the get_vit aliases too
    size = base_name.split('_')[1]   # 'vit_base_patch16_224...' -> 'base', 'vit_b_16' -> 'b'
    depths = {'t': 12, 'tiny': 12, 's': 12, 'small': 12, 'medium': 12,
              'b': 12, 'base': 12, 'l': 24, 'large': 24, 'h': 32, 'huge': 32,
              'giant': 40, 'gigantic': 48}
    if size not in depths:
        raise ValueError(f"Can't infer ViT depth from '{base_name}', set exp_params['vit_depth']")
    return depths[size]
def get_optimization_groups(model_name,exp_params):
    last_stage = None
    if exp_params['use_opt_groups'] == False:
        return None, None
    if 'resnet' in model_name:
        decay = exp_params.get('lr_decay', 0.9)
        define_optimization_groups = [
            {'names': ['vision_model.conv1','vision_model.bn1'],'lr': exp_params['lr_backbone']*decay**4, 'lr_name': 'lr_stem'},
            {'names': ['layer1'],'lr': exp_params['lr_backbone']*decay**3, 'lr_name': 'lr_1'},
            {'names': ['layer2'],'lr': exp_params['lr_backbone']*decay**2, 'lr_name': 'lr_2'},
            {'names': ['layer3'],'lr': exp_params['lr_backbone']*decay, 'lr_name': 'lr_3'},
            {'names': ['layer4'],'lr': exp_params['lr_backbone'], 'lr_name': 'lr_4'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
        ] # or None or other configurations fo other models
    elif 'FiveStageResidualStridedConvNet' in model_name:
        define_optimization_groups = [
            {'names': ['stem', 'stages.0'], 'lr': exp_params['lr_backbone']*0.1*0.1, 'lr_name': 'lr_1'},   # ~ conv1/bn1/layer1
            {'names': ['stages.1'],         'lr': exp_params['lr_backbone']*0.1, 'lr_name': 'lr_2'},   # ~ layer2
            {'names': ['stages.2'],         'lr': exp_params['lr_backbone']*0.5, 'lr_name': 'lr_3'},   # ~ layer3
            {'names': ['stages.3', 'stages.4'], 'lr': exp_params['lr_backbone'], 'lr_name': 'lr_4'},  # ~ layer4
            {'names': ['head', 'projector'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_classifier'},
        ]
    elif 'convnext' in model_name:
        decay = exp_params.get('lr_decay', 0.8)
        lrb = exp_params['lr_backbone']
        define_optimization_groups = [
            {'names': ['vision_model.features.0'], 'lr': lrb*decay**4, 'lr_name': 'lr_stem'},
            {'names': ['vision_model.features.1'], 'lr': lrb*decay**3, 'lr_name': 'lr_1'},
            {'names': ['vision_model.features.2', 'vision_model.features.3'], 'lr': lrb*decay**2, 'lr_name': 'lr_2'},
            {'names': ['vision_model.features.4', 'vision_model.features.5'], 'lr': lrb*decay,    'lr_name': 'lr_3'},
            {'names': ['vision_model.features.6', 'vision_model.features.7',
                    'vision_model.final_norm'], 'lr': lrb, 'lr_name': 'lr_4'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'}, #classifier here matches also vision_model.classifier, but we keep it separate for logging purposes
            # but it is ok since if one name matches more than on elayer only the first instance counts
        ]
        '''
        recommended for pre-training: 
        lr_backbone: 1e-4 , range: 5e-5 to 2e-4
        lr_decay 0.75, 0.65 to 0.85
        lr_classifier_head: 1e-3, range: 5 to 10xbackbone
        weight_decay: 0.05, range: 0.02 to 0.1

        These assume an effective batch of roughly 128 to 256. 
        Learning rate scales roughly linearly with batch size, so halve lr_backbone if your batch is about 64, and so on.

        '''
    elif 'efficientnet_v2' in model_name:
        decay = exp_params.get('lr_decay', 0.8)
        lrb = exp_params['lr_backbone']
        p = 'vision_model.features.'

        # _s has 6 stages, _m and _l have 7; the last entry is the final 1x1 conv
        n_stages = 6 if '_v2_s' in model_name else 7
        last = [f'{p}{i}' for i in range(6, n_stages + 2)]  # stage6 [, stage7] + final conv

        define_optimization_groups = [
            # stem + stage1 both run at /2, stage1 is stride 1
            {'names': [f'{p}0', f'{p}1'],       'lr': lrb*decay**4, 'lr_name': 'lr_stem'},
            {'names': [f'{p}2'],                'lr': lrb*decay**3, 'lr_name': 'lr_1'},   # /4
            {'names': [f'{p}3'],                'lr': lrb*decay**2, 'lr_name': 'lr_2'},   # /8
            {'names': [f'{p}4', f'{p}5'],       'lr': lrb*decay,    'lr_name': 'lr_3'},   # /16, stage5 is stride 1
            {'names': last,                     'lr': lrb,          'lr_name': 'lr_4'},   # /32
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
        ]
    elif 'swin' in model_name:
        decay = exp_params.get('lr_decay', 0.8)
        lrb = exp_params['lr_backbone']
        p = 'vision_model.features.'
        define_optimization_groups = [
            {'names': [f'{p}0'],          'lr': lrb*decay**4, 'lr_name': 'lr_stem'},
            {'names': [f'{p}1'],          'lr': lrb*decay**3, 'lr_name': 'lr_1'},
            {'names': [f'{p}2', f'{p}3'], 'lr': lrb*decay**2, 'lr_name': 'lr_2'},
            {'names': [f'{p}4', f'{p}5'], 'lr': lrb*decay,    'lr_name': 'lr_3'},
            {'names': [f'{p}6', f'{p}7',
                       'vision_model.final_norm'], 'lr': lrb, 'lr_name': 'lr_4'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
        ]
    elif model_name.startswith(('vit_', 'deit')):
        decay = exp_params.get('lr_decay', 0.8)
        lrb = exp_params['lr_backbone']
        p = 'vision_model.'

        base_name, _, suffix = model_name.partition('_layer')
        depth = exp_params.get('vit_depth') or _vit_depth(base_name)   # full depth, before truncation
        if suffix:  # must mirror the cut in get_vit
            depth = max(1, int(suffix) * depth // 4)

        stem = [f'{p}cls_token', f'{p}reg_token', f'{p}pos_embed', f'{p}patch_embed.', f'{p}norm_pre.']
        top_extra = [f'{p}fc_norm.', f'{p}attn_pool.']
        head_names = ['classifier']
        if suffix:
            # get_vit replaces model.norm with a freshly initialised LayerNorm -> train it at head lr
            head_names.insert(0, f'{p}norm.')
        else:
            top_extra.insert(0, f'{p}norm.')

        if exp_params.get('llrd_per_block', False):
            # classic ViT layer-wise decay: one group per block (lr_decay ~0.65 for B, ~0.75 for L)
            chunks = [[i] for i in range(depth)]
        else:
            # 4 equal stages, same convention as the CNN entries (lr_decay per stage)
            n = min(4, depth)
            bounds = [i * depth // n for i in range(n + 1)]
            chunks = [list(range(bounds[i], bounds[i + 1])) for i in range(n)]

        k = len(chunks)
        define_optimization_groups = [{'names': stem, 'lr': lrb * decay**k, 'lr_name': 'lr_stem'}]
        for j, blocks in enumerate(chunks):
            names = [f'{p}blocks.{b}.' for b in blocks]
            if j == k - 1:
                names += top_extra
            define_optimization_groups.append(
                {'names': names, 'lr': lrb * decay**(k - 1 - j), 'lr_name': f'lr_{j + 1}'})
        define_optimization_groups.append(
            {'names': head_names, 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'})
    elif model_name.startswith('ConvEncoder'):          # e.g. ConvEncoder_dae_128, ConvEncoder_cvae_128
        decay = exp_params.get('lr_decay', 0.8)
        lrb = exp_params['lr_backbone']

        pretraining_mode, input_size = _process_name(model_name)
        specific_path = mapping_conv_encoder.get((pretraining_mode, input_size), None)
        config_path = os.path.join(SOURCE_PATH, pretraining_mode, os.path.dirname(os.path.dirname(specific_path))) 
        n_channels, bps = _load_preprocessing_params(config_path, ('channels','blocks_per_stage'), 'model.init_args')
        n_stages = len(n_channels)
        
        p = 'vision_model.'                               # attribute holding the ConvEncoderWrapper
        e = f'{p}encoder.'

        # ConvEncoder.body = per stage: `bps` residual blocks, then a Downsample (none after the last stage)
        stage_names, idx = [], 0
        for s in range(n_stages):
            n = bps + (1 if s < n_stages - 1 else 0)
            stage_names.append([f'{e}body.{i}.' for i in range(idx, idx + n)])   # trailing '.' matters
            idx += n
        stage_names[0] = [f'{e}stem.'] + stage_names[0]
        stage_names[-1] += [f'{e}out_norm.', f'{e}head.', f'{p}extraction_module.']   # CVAE: to_stats

        define_optimization_groups = [
            {'names': names, 'lr': lrb * decay ** (n_stages - 1 - j), 'lr_name': 'lr_stem' if j == 0 else f'lr_{j}'}
            for j, names in enumerate(stage_names)
        ]
        define_optimization_groups.append(
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'})
        last_stage = ['vision_model.encoder.body.12.', 'vision_model.encoder.body.13.', 'vision_model.encoder.out_norm.', 'vision_model.encoder.head.', 'vision_model.extraction_module.']# lr4
    return define_optimization_groups, last_stage
#['classifier','vision_model.features.6','vision_model.features.7','vision_model.final_norm'], #['all'],#['classifier','layer4'],#['all','classifier'], #Update it for every model
    #['stages.3', 'stages.4', 'head', 'projector', 'classifier']


def get_input_modality(name):
    if name=='mixed_view':
        return ['X_crop', 'digit_full', 'digit_crop', 'digit', 'digit', 'digit', 'text_full', 'text_crop', 'text', 'text', 'text']
    elif name=='window_view':
        return ['X_window' for _ in range(3)]+['text_window' for _ in range(3)]+ ['digit_window' for _ in range(3)]
    elif name=='window_view_minimal':
        return ['X_window']+['text_window']+ ['digit_window']
    elif name=='window_view_2':
        n=2
        return ['X_window' for _ in range(n)]+['text_window' for _ in range(n)]+ ['digit_window' for _ in range(n)]
    elif name=='window_no_text':
        n=3
        return ['X_window' for _ in range(n)]+ ['digit_window' for _ in range(n)]
    elif name=='window_debug_aug':
        return ['X_window','_X_window']+ ['digit_window' , '_digit_window']+ ['text_window', '_text_window']
    elif name=='window_3x_2d_1t':
        return ['X_window' for _ in range(3)]+['digit_window' for _ in range(2)]+ ['text_window']
    elif name=='patch_view':
        n_x=20
        n_d=10
        n_t=6
        return ['X_patch' for _ in range(n_x)]+ ['digit_patch' for _ in range(n_d)]+['text_patch' for _ in range(n_t)]
    elif name=='patch_view_balanced':
        n_x=10
        n_d=10
        n_t=10
        return ['X_patch' for _ in range(n_x)]+ ['digit_patch' for _ in range(n_d)]+['text_patch' for _ in range(n_t)]


#Set hyperparameters / metadata
def set_automatic_hyperparameters(exp_params):
    bar = "=" * 80

    problem = exp_params.get('problem', 'PD')
    if problem=='handedness':
        exp_params['class_col'] = 'lateralite' 
        exp_params['merge_shards'] = True #merge train and val since you are using the PD pre_training shards as trainign set
        exp_params['filter_missing'] = 'all'
        exp_params['censor_time'] = 'all'
        if exp_params['balanced_data'] == False:
            print("\n[WARNING] You are not balancing, you are training on a large dataset -> setting intermediate validation steps\n")
            exp_params['val_check_interval'] = 0.3 #None or float between 0 and 1, if None validation is done at the end of each epoch, if float validation is done every val_check_interval fraction of an epoch
            exp_params['align_train_metrics_to_val'] = True
        else:
            exp_params['val_check_interval'] = None
            exp_params['align_train_metrics_to_val'] = False
        '''
        Split: train -> 0.0    38559 , 1.0     3316
        Split: val -> 0.0    8263 , 1.0     710
        '''
    elif problem=='PD':
        exp_params['class_col'] = 'diag_park_final1_quest'
    elif problem=='unsupervised_pretraining':
        exp_params['class_col'] = 'no_label'

    if exp_params.get('from_representation_file', None):
        exp_params['representation_file_path'] = get_representation_file_path(exp_params['from_representation_file'])
        exp_params['model'] = 'identity'  # override model to identity since we are loading representations from a file
        exp_params['debug_imgs'] = False
        exp_params['representation_dim'] = read_rep_dim(exp_params['representation_file_path'])
        print(f"Using representation file: {exp_params['representation_file_path']}", flush=True)
        print("[INFO] Overriding model to 'identity' since we are loading representations from a file", flush=True)
    
    if exp_params.get('label_smoothing', 0.0)!=0.0:
        exp_params['num_classes'] = 2
        msg = (f"Label smoothing is only compatible with the cross entropy loss ->"
               f"setting num_classes=2")
        print(f"\n{bar}\n>>> [NUM CLASSES] {msg}\n{bar}\n", flush=True)
    
    scale_lr_with_batch = exp_params.get('scale_lr_with_batch_size', False)
    if scale_lr_with_batch:
        effective_batch_size = exp_params['batch_size']*exp_params.get('accumulate_grad_batches', 1)
        ref_size = 256
        exp_params['lr_backbone'] *= effective_batch_size / ref_size
        exp_params['lr_classifier_head'] *= effective_batch_size / ref_size
        msg = (f"Scaling learning rates with effective batch size {effective_batch_size} (base {ref_size}): "
                f"lr_backbone={exp_params['lr_backbone']}, lr_classifier_head={exp_params['lr_classifier_head']}")
        print(f"\n{bar}\n>>> [LR SCALING] {msg}\n{bar}\n", flush=True)
    
    if exp_params.get('lora_tuning', False):
        exp_params['layers_to_unfreeze'] = ['classifier']
        exp_params['use_opt_groups'] = False

    if exp_params.get('feature_extraction', False):
        exp_params['debug']=True
    
    if exp_params.get('grouped', False) or exp_params.get('pre_training', False) or problem=='unsupervised_pretraining':
        exp_params['balance_validation'] = False
        exp_params['balanced_data'] = False
        exp_params['use_balanced_weights'] = False
    
    if exp_params.get('pre_training', False) or problem=='unsupervised_pretraining':
        exp_params['filter_missing'] = 'all'
        exp_params['censor_time'] = 'all'
    elif problem not in ['handedness']:
        exp_params['val_check_interval'] = None #force epoch level logging if i am not doing pre_trainign
        exp_params['align_train_metrics_to_val'] = False
    
    exp_params['num_channels'] = 1 if exp_params['to_grayscale'] else 3
    if isinstance(exp_params['data_modality'],list):
        exp_params['num_tiles'] = len(exp_params['data_modality'])

    huggingface_transform=False
    if 'model' in exp_params:
        huggingface_transform=True if exp_params['model'] in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
    exp_params['huggingface_transform'] = huggingface_transform

    if exp_params['debug']:
        exp_params['custom_transform'] = 'pad_resize_pil' 
    return exp_params
