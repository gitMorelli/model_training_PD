import torch
import torchvision.transforms as T
import time
from .torch_models import *
from .timm_models import *
from .custom_backbones import *
from .heads import *
from .transforms.transforms import simple_resize_transform


#Model loading
def get_model(name="resnet50", mode='classification head', pretrained=True,checkpoint_path=None, is_lora=False,
              **kwargs):
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
    elif name.startswith('convnext'):
        if is_lora:
            model = get_convnext_lora(name, mode, pretrained, **kwargs)
            transform = get_convnext_lora_transforms(model=model, **kwargs)
        else:
            model = get_convnext(name, mode, pretrained, **kwargs)
            transform = get_convnext_transforms(**kwargs)
    elif name.startswith('efficientnet_v2'):
        model = get_efficientnet_v2(name, mode, pretrained, **kwargs)
        transform = get_efficientnet_v2_transforms(**kwargs)
    elif name.startswith('vit'):
        if is_lora:
            model = get_vit_lora(name, mode, pretrained, **kwargs)
            transform = get_vit_lora_transforms(model=model, **kwargs)
        else:
            raise ValueError(f"Model {name} is not supported. Use lora=True for vit models.")
    elif name.startswith('clip-vit'):
        model = get_clip_vit(name, **kwargs)
        transform = get_clip_vit_transforms(name, **kwargs) 
    elif name.startswith('swin'):
        model = get_swin(name, mode, pretrained, **kwargs)
        transform = get_swin_transforms(**kwargs)
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
        class_weight=kwargs.get('class_weight',None)
        return LogisticRegression(max_iter=max_iter, random_state=42, penalty=penalty, C=C, solver=solver, class_weight=class_weight)
    elif name=='logreg_cv':
        from sklearn.linear_model import LogisticRegressionCV
        penalty=kwargs.get('penalty','l2')
        Cs=kwargs.get('Cs',10)
        solver=kwargs.get('solver','lbfgs')
        max_iter=kwargs.get('max_iter',5000)
        class_weight=kwargs.get('class_weight',None)
        return LogisticRegressionCV(max_iter=max_iter, random_state=42, penalty=penalty, Cs=Cs, solver=solver, class_weight=class_weight)
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
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            num_leaves=kwargs.get('num_leaves', 31), #Maximum number of leaves in one tree
            min_child_samples=kwargs.get('min_child_samples', 20),#Minimum number of data samples per leaf
            subsample=kwargs.get('subsample', 0.8), #Randomness in row 
            colsample_bytree=kwargs.get('colsample_bytree', 0.8), #and feature sampling respectively.
            reg_alpha=kwargs.get('reg_alpha', 1.0), # L1 regularization
            reg_lambda=kwargs.get('reg_lambda', 1.0), # L2 regularization
            random_state=42,
            n_jobs=-1,
            min_split_gain=kwargs.get('min_split_gain', 0.01),  # Minimum gain to make a split
            class_weight=kwargs.get('class_weight', None)
        )
    elif name=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=kwargs.get('n_estimators', 100), learning_rate=kwargs.get('learning_rate', 0.1), 
                             max_depth=kwargs.get('max_depth', 3), random_state=42,
                             scale_pos_weight=kwargs.get('scale_pos_weight', 1), eval_metric=kwargs.get('eval_metric', 'logloss'), 
                             max_delta_step=kwargs.get('max_delta_step', 0), min_child_weight=kwargs.get('min_child_weight', 1))
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

def unfreeze_layers(model, layer_names=['all'], keep_lora=True):
    if len(layer_names) == 0:
        layer_names = ['frozen']
    for name, param in model.named_parameters():
        if layer_names[0] == 'all' or any(l in name for l in layer_names):
            param.requires_grad = True
        elif keep_lora and 'lora_' in name:
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

#Load from custom checkpoint
def load_ln_checkpoint(full_model, custom_pre_trained_weights, load_full_model=False):
    if custom_pre_trained_weights is not None:
        checkpoint = torch.load(custom_pre_trained_weights, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        prefix = "model."  # set to "" if there's no wrapper prefix

        if load_full_model:
            # Assume head also matches: load everything, just strip the prefix
            weights_to_load = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            # Only load backbone weights
            weights_to_load = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k[len(prefix):].startswith("vision_model.")
            }

        missing, unexpected = full_model.load_state_dict(weights_to_load, strict=False)

        assert not unexpected, f"Unexpected keys: {unexpected}"

        if load_full_model:
            assert not missing, f"Some weights didn't load: {missing}"
        else:
            assert all(not k.startswith("vision_model.") for k in missing), \
                f"Some backbone weights didn't load: {[k for k in missing if k.startswith('vision_model.')]}"

        scope = "full model" if load_full_model else "backbone"
        print(f"Loaded weights into the {scope} from {custom_pre_trained_weights}.")

    return full_model

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

