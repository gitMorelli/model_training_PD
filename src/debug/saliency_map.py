import tarfile
import time
import io
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import time
import webdataset as wds
import glob
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import torchvision.utils as vutils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassRecall
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import re
import signal
import sys
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F
import matplotlib.pyplot as plt


from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, melt_df
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide
from src.utils.training_utils import LitModel

#PATHS
SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
CSV_LOAD_PATH = os.path.join("/mnt/beegfs02/scratch/a_morelli/model_training/handedness/",
                             "resnet18_model_results/checkpoints/v_30/merged_statistics_w_predictions_w_original.csv"
)
MODEL_SPECIFIC_SAVE_PATH = os.path.dirname(CSV_LOAD_PATH)
data_folder = "all_no_grids_png_whitebg" 
#data_folder = "all_full_sentences_png_whitebg" 
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

#Model definition
MODEL = 'resnet18' #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
SAVE_PATH = f"/home/a_morelli/vscode_projects/model_training/data/saliency_maps/{MODEL}"

#clip-vit-large-patch14, clip-vit-large-patch14-inter
huggingface_transform=True if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
transform_override = True #if true overrides the transform defined for the model with ta custom one
CLASSIFICATION_HEAD = 'linear' #'MLPClassifier1'#'MLPClassifier1' # 'linear', 'regularized_linear', 'MLPClassifier1'
PARAMS = {
    'dropout': 0.2,
    'hidden_sizes': [32],
    'with_input_norm': 'batch_norm'
}
batch_size = 32

EXPERIMENT_NAME = f"{MODEL}_{data_folder}"
RESULTS_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
CHECKPOINT_PATH = os.path.join(RESULTS_PATH, "checkpoints")
checkpoint_to_load='v_30/best-epoch=09-val_loss=0.69.ckpt'#best-epoch=55-val_loss=0.91.ckpt'#best.ckpt , None last.ckpt

DEBUG_IMGS = False
SEED=42
DATA_MODALITY = 'number_random' # 'all' or one from DATA_MODALITIES
NUM_tiles = 1
MODE_FOR_SALIENCY_MAP_GENERATION = 'handedness_singlemode' # 'handedness_multimode', 'handedness_singlemode', 
DATA_MODALITIES = ['hand', 'number_random', 'X']

BALANCED_DATA = True
USE_BALANCED_WEIGHTS = False
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0
THRESHOLD_NUM = 1
PREDICT_ON_TRAIN = True

metadata = {
    'model': MODEL,
    'checkpoint': checkpoint_to_load,
    'data_modality': DATA_MODALITY,
    'mode_for_saliency_map_generation': MODE_FOR_SALIENCY_MAP_GENERATION,
    'csv_load_path': CSV_LOAD_PATH,
    'source_pattern': SOURCE_PATTERN,
}

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


def get_result_df(outputs):
    # 2. Concatenate all batch outputs into unified tensors
    all_probs = torch.cat([batch["probs"] for batch in outputs])
    all_preds = torch.cat([batch["preds"] for batch in outputs])
    all_labels = torch.cat([batch["labels"] for batch in outputs])
    all_subjects = [sid for batch in outputs for sid in batch["subject_id"]]
    all_questionnaires = [q for batch in outputs for q in batch["questionnaire"]]

    #create a dataframe with the subject id, questionnaire, true label, predicted label and probabilities
    results_df = pd.DataFrame({
        "subject_id": all_subjects,
        "questionnaire": all_questionnaires,
        "true_label": all_labels.numpy(),
        "predicted_label": all_preds.numpy(),
        "probability_0": all_probs[:, 0].numpy(),
        "probability_1": all_probs[:, 1].numpy() 
    })
    return results_df, all_probs, all_preds, all_labels

def one_channel_gradcam(model, rgb_img, input_tensor,out_path, device):
    ''' input tensor should be normalized and of shape [1, 3, H, W]
    rgb image in [0,1] for the overlay (NOT normalized)'''
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True) 

    target_layers = [model.vision_model.layer4[-1]]

    print(f"Generating Grad-CAM for input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")

    cam = GradCAM(model=model, target_layers=target_layers)

    # targets=None -> uses the highest-scoring class; or pick a class:
    targets = [ClassifierOutputTarget(0)]   

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # [H, W]
    visualization = show_cam_on_image(rgb_img.astype(np.float32),
                                    grayscale_cam, use_rgb=True)

    Image.fromarray(visualization).save(out_path)

def multi_view_gradcam(model, sample, rgb_imgs, device,out_dir="gradcam_output"):
    '''
    rgb_imgs must be the three views individually, un-normalized in [0,1], 
    in the same view order as sample — so view i's CAM lands on view i's image.
    '''
    model.eval()

    target_layer = model.vision_model.layer4[-1]

    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations["v"] = out
        out.register_hook(lambda grad: gradients.__setitem__("v", grad))

    h = target_layer.register_forward_hook(fwd_hook)

    # input: (1, n, 3, 224, 224)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)           # ensures the graph is built even if params were frozen
    print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")

    output = model(input_tensor)                # (1, num_classes)
    class_idx = output.argmax(dim=1)

    model.zero_grad()
    output[0, class_idx].backward()
    h.remove()

    a = activations["v"]                        # (n, C, h, w)
    g = gradients["v"]                          # (n, C, h, w)

    weights = g.mean(dim=(2, 3), keepdim=True)  # (n, C, 1, 1)
    cam = F.relu((weights * a).sum(dim=1))      # (n, h, w)

    cam = F.interpolate(cam.unsqueeze(1), size=(224, 224),
                        mode="bilinear", align_corners=False).squeeze(1)
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (
        cam.amax(dim=(1, 2), keepdim=True)
        - cam.amin(dim=(1, 2), keepdim=True) + 1e-8)
    cam = cam.detach().cpu().numpy()            # (n, 224, 224)

    # rgb_imgs: list of the n un-normalized views, each (224, 224, 3) in [0, 1]
    for i in range(cam.shape[0]):
        vis = show_cam_on_image(rgb_imgs[i].astype(np.float32), cam[i], use_rgb=True)
        Image.fromarray(vis).save(os.path.join(out_dir,f"gradcam_view{i}.jpg"))

def inspect_conv_layer(model,out_path='conv_1.png'):
    w = model.vision_model.conv1.weight.data.clone().cpu()   # [64, 3, 7, 7] for resnet50
    w = (w - w.min()) / (w.max() - w.min())      # normalize to [0,1]

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(w[i].permute(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)

def inspect_activations(model,input_tensor, device,out_path="layer1_features.png"):
    activations = {}
    input_tensor = input_tensor.to(device)

    def hook_fn(name):
        def fn(module, inp, out):
            activations[name] = out.detach().cpu()
        return fn

    # register on whatever layers interest you
    h1 = model.vision_model.layer1.register_forward_hook(hook_fn("layer1"))
    h4 = model.vision_model.layer4.register_forward_hook(hook_fn("layer4"))

    with torch.no_grad():
        _ = model(input_tensor)

    h1.remove(); h4.remove()

    # visualize the first 16 channels of layer1's output
    act = activations["layer1"][0]   # [C, H, W]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(act[i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)

def show_hookable_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name, tuple(module.weight.shape))

def prepare_from_shards(n_subjects=5, mode='handedness_multimode', selected_modality='number_random'):
    shard_path = os.path.join(SOURCE_PATTERN,"train")
    csv_data = pd.read_csv(CSV_LOAD_PATH)
    csv_data = csv_data[csv_data['split'] == 'train']
    #for the rows in which modality_type==x map it to X
    csv_data['modality_type'] = csv_data['modality_type'].replace({'x': 'X'})

    def process_path(path):
        in_tar_path = path.split('/')[1]
        questionnaire = in_tar_path.split('.')[1]
        modality = in_tar_path.split('.')[2]
        subject_id = in_tar_path.split('.')[0]
        return subject_id,questionnaire, modality

    def reorder_paths(dict_of_subjects, mode='handedness_multimode',selected_modality='number_random'):
        new_dict = {}  # Filter out subjects with no paths
        for subject in dict_of_subjects:
            new_dict[subject] = {}
            if mode == 'handedness_multimode': 
                for q in range(1,14):
                    q_paths = [path for path in dict_of_subjects[subject] if process_path(path)[1] == 'q'+str(q)]
                    #sort the paths by modality ('hand', 'number_random', 'X') #see data_loading_utils
                    new_q_paths = []
                    modalities=DATA_MODALITIES
                    for modality in modalities:
                        new_q_paths.append(None)
                        if len(q_paths) == 0:
                            new_dict[subject][q]=None 
                            break
                        for path in q_paths:
                            if process_path(path)[2] == modality:
                                new_q_paths[-1]=path
                                break
                    new_dict[subject][q] = new_q_paths
            elif mode == 'handedness_singlemode':
                for q in range(1,14):
                    q_paths = [path for path in dict_of_subjects[subject] if process_path(path)[1] == 'q'+str(q)]
                    new_q_paths = []
                    new_q_paths.append(None)
                    if len(q_paths) == 0:
                        new_dict[subject][q]=None 
                        continue
                    for path in q_paths:
                        if process_path(path)[2] == selected_modality:
                            new_q_paths[-1]=path
                            break
                    new_dict[subject][q] = new_q_paths
        return new_dict
            
    def return_images(list_of_paths):
        '''
        takes a list of paths and extract the images saving them as a list of rgb images
        '''
        if list_of_paths is None:
            return list_of_paths
        img_list = []
        for member in list_of_paths:
            if member is None:
                img_list.append(None)
                continue
            shard=os.path.join(shard_path,member.split('/')[0]+".tar")
            subject_id = member.split('/')[1].split('.')[0]
            #print(shard)
            '''with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("Corresponding rows in CSV:")
                print(csv_data[csv_data['subject_id']==subject_id])'''
            with tarfile.open(shard) as tar:
                '''for m in tar.getmembers():
                    if subject_id in m.name:
                        print(f"Extracting {m.name} from {shard}")'''
                f = tar.extractfile(member.split('/')[1])
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                img_list.append(img.copy())
        return img_list
    def random_samples_from_shards(n_random,list_of_ids,modality,questionnaire):
        #this function enables selecting specific ids and showing the images from the shards directly
        filtered_data = csv_data.copy()
        try:
            if list_of_ids: #if none i can sample any subject
                filtered_data = filtered_data[filtered_data['subject_id'].isin(list_of_ids)]
            if modality:
                #if modality is not none filter for the specified modality
                filtered_data = csv_data[csv_data['modality_type'] == modality]
            if questionnaire:
                #if questionnaire is not none filter for the specified questionnaire
                filtered_data = filtered_data[filtered_data['questionnaire'].isin(questionnaire)]
            unique_ids = filtered_data['subject_id'].unique()
            #select n_random random samples from the unique ids
            selected_ids = np.random.choice(unique_ids, size=min(n_random, len(unique_ids)), replace=False)
            
            #i return all available files for a spcific subject, i will filter and order later
            images_per_id={}
            for subject_id in selected_ids:
                subset = filtered_data[filtered_data['subject_id'] == subject_id]
                images_per_id[subject_id] = [f"{row['shard_name'].split('.')[0]}/{row['subject_id']}.{row['questionnaire']}.{row['modality_type']}.png" for index, row in subset.iterrows()]
            
        except Exception as e:
            print(f"Error filtering data: {e}")
            images_per_id = {}
        return images_per_id

    images_per_id = random_samples_from_shards(n_random=n_subjects,list_of_ids=None, modality=None, questionnaire=None)
    #print(images_per_id)
    images_per_id_and_q = reorder_paths(images_per_id, mode=mode)
    #print(images_per_id_and_q)
    #assert 1==0, "Debugging: stop execution after preparing images_per_id_and_q"
    dict_of_images = {}
    for subject_id in images_per_id_and_q:
        dict_of_images[subject_id] = {}
        for q in images_per_id_and_q[subject_id]:
            dict_of_images[subject_id][q] = return_images(images_per_id_and_q[subject_id][q])
    return dict_of_images


input_size = 224
custom_transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                        std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
        ]
    )
transform_wo_norm = T.Compose(
        [
            T.Resize((input_size, input_size)),
            #rescale to [0,1] for visualization but keep rgb
        ]
)
AUGMENTATION_TRANSFORM = T.Compose([
                #ResizeLongestSide(448),
                T.RandomCrop(
                    112, 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ])

def main(clear_folder=True,copy_results_to_model_folder=True,run_layer_inspector=True):
    args = get_args()
    transform = None
    apply_augmentation = True
    invert_color=True

    if clear_folder:
        #clear the gradcam_output folder if it exists
        gradcam_output_path = os.path.join(SAVE_PATH,"gradcam_output")
        model_specific_output_path = os.path.join(MODEL_SPECIFIC_SAVE_PATH, "saliency_maps_and_activations")
        if os.path.exists(gradcam_output_path):
            shutil.rmtree(gradcam_output_path)
            print(f"Cleared existing folder: {gradcam_output_path}")
        else:
            print(f"No existing folder to clear: {gradcam_output_path}")
        if os.path.exists(model_specific_output_path):
            shutil.rmtree(model_specific_output_path)
            print(f"Cleared existing folder: {model_specific_output_path}")
        else:
            print(f"No existing folder to clear: {model_specific_output_path}")

    #fix all the seeds for reproducibility 
    torch.manual_seed(SEED)
    random.seed(SEED)
    #with lightning 
    L.seed_everything(SEED, workers=True)

    if NUM_tiles > 1 and DATA_MODALITY == 'all':
        print("Warning: Data modality = 'all' and NUM_tiles>1 are incompatible ")
        return 


    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = AUGMENTATION_TRANSFORM
    else:
        augmentation_transform = None
    if transform_override:
        transform = custom_transform


    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lit_model = LitModel.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, checkpoint_to_load), write_log=False)
    #(write_log=None,model=model, num_0=1, num_1=1, num_classes=num_classes, example_input_array=example_input_array)
    lit_model.eval()
    lit_model.to(device) 
    lit_model.freeze()  # optional; we'll re-enable grads for gradcam below

    # the underlying resnet — adapt the attribute name to your code
    model = lit_model.model   # or lit_model.backbone, etc.

    #print the model architecture
    '''print("Model architecture:")
    for name, param in model.named_parameters():
        print(name, param.requires_grad)'''
    '''#show the forward method of the model
    print("Model forward method:")
    print(model.forward)'''

    dict_of_images_and_q = prepare_from_shards(n_subjects=5, mode=MODE_FOR_SALIENCY_MAP_GENERATION, selected_modality=DATA_MODALITY)
    for subject_id in dict_of_images_and_q:
        print(f"Processing subject {subject_id}...")
        
        for questionnaire in dict_of_images_and_q[subject_id]:
            print(f"     questionnaire {questionnaire}...")
            transformed_images=[]
            reference_images=[]
            list_of_images = dict_of_images_and_q[subject_id][questionnaire]
            if list_of_images is None:
                print(f"No images found for subject {subject_id} and questionnaire {questionnaire}. Skipping.")
                continue
            #print(len(list_of_images))
            #continue
            for source_img in list_of_images:
                if source_img is None:
                    source_img = Image.new('RGB', (input_size, input_size), (255, 255, 255))  # create a white image
                #apply the augmentation transform if specified
                img = augmentation_transform(source_img)
                if invert_color:
                    img = ImageOps.invert(img)
                rgb_img = transform_wo_norm(img)
                rgb_img = np.array(rgb_img).astype(np.float32) / 255.0
                img_tensor = transform(img)
                transformed_images.append(img_tensor)
                reference_images.append(rgb_img)

            if MODE_FOR_SALIENCY_MAP_GENERATION == 'handedness_multimode':
                sample = torch.stack(transformed_images).unsqueeze(0)
                out_dir = os.path.join(SAVE_PATH,"gradcam_output", f"{subject_id}",f"q{questionnaire}")
                os.makedirs(out_dir, exist_ok=True)
                multi_view_gradcam(model, sample, reference_images, out_dir=out_dir, device=device)
            elif MODE_FOR_SALIENCY_MAP_GENERATION  == 'handedness_singlemode':
                sample = transformed_images[0].unsqueeze(0)
                out_dir = os.path.join(SAVE_PATH,"gradcam_output", f"{subject_id}")
                os.makedirs(out_dir, exist_ok=True)
                one_channel_gradcam(model, reference_images[0], sample, device=device,out_path=os.path.join(out_dir,f"q{questionnaire}_gradcam.jpg"))
            if run_layer_inspector:
                out_dir = os.path.join(SAVE_PATH,"layer_inspection", f"{subject_id}")
                os.makedirs(out_dir, exist_ok=True)
                inspect_activations(model, sample, device=device, out_path=os.path.join(out_dir,f"q{questionnaire}_layer_features.png"))
    if run_layer_inspector:
        #inspect the first conv layer
        out_dir = os.path.join(SAVE_PATH,"layer_inspection")
        inspect_conv_layer(model, out_path=os.path.join(out_dir,"conv_1.png"))
        '''#inspect the activations of layer1
        sample = transformed_images[0].unsqueeze(0)
        inspect_activations(model, sample, out_path=os.path.join(SAVE_PATH,"layer1_features.png"))'''
    if copy_results_to_model_folder:
        #copy the gradcam_output folder to the model results folder
        dest_dir = os.path.join(MODEL_SPECIFIC_SAVE_PATH, "saliency_maps_and_activations")
        shutil.copytree(SAVE_PATH, dest_dir, dirs_exist_ok=True)

    

if __name__ == "__main__":
    main()
    