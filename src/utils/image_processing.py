import cv2
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import os
from skimage.filters import threshold_otsu 
from scipy import ndimage

def convert_background_to_white(image):
    img = image.copy()
    #convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. Threshold the image to isolate the white squares
    # This turns anything bright into pure white (255) and everything else to black (0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # 3. Find the external contours of the white squares
    # RETR_EXTERNAL ensures we only grab the outer borders of the squares, ignoring the symbols inside
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Create a blank black mask of the same size
    mask = np.zeros_like(img)

    # 5. Fill the detected square contours with pure white on our mask
    # Because we use cv2.FILLED, the entire inside of the square becomes white on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 6. Create the final image
    result = img.copy()
    
    # Where the mask is 0 (which means it's the original outer background), change it to white (255)
    result[mask == 0] = 255

    return result


def get_tiles(image,coords,num_tiles):
    # Create a copy to avoid modifying the original image array
    output_img = image.copy()
    h, w = output_img.shape[:2]
    #identify all the tiles defined by the coordinates
    #build the list of UL coordinates
    new_coords = []
    x_coords = [0]+sorted(list(coords[0, :]))+[w]
    y_coords = [0]+sorted(list(coords[1, :]))+[h]
    size = len(x_coords)-1
    processed_tiles=0
    for i in range(size):
        for j in range(size):
            processed_tiles+=1
            if processed_tiles<=num_tiles:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),processed_tiles])
            else:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),-1]) #empty tiles
    return new_coords


def recolor_border_via_profiles(image, coords, black_tolerance=5):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, black_tolerance, 255, cv2.THRESH_BINARY)

    
    for tile in coords:
        x1, y1, x2, y2 = tile[0]
        tile_num = tile[1]
        tile_img = thresh[y1:y2, x1:x2]
        # Compute the sum of pixel values along rows and columns
        x_profile = np.sum(tile_img, axis=0)
        y_profile = np.sum(tile_img, axis=1)

        # 4. Find the indices where the profiles are greater than 0
        # This means there is at least one non-black pixel in that row/column
        x_content_indices = np.where(x_profile > 0)[0]
        y_content_indices = np.where(y_profile > 0)[0]

        # Handle the edge case where the image is entirely black
        if len(x_content_indices) == 0 or len(y_content_indices) == 0 or tile_num==-1:
            #set the tile to white
            image[y1:y2, x1:x2] = 255
            continue
        
        # 5. Identify the bounding box coordinates
        # The first and last indices represent the edges of the core content
        x_min, x_max = x_content_indices[0], x_content_indices[-1]
        y_min, y_max = y_content_indices[0], y_content_indices[-1]

        # 7. Create a white background of the original image size
        white_background = np.full_like(img[y1:y2,x1:x2], 255)

        # 8. Paste the core image into the exact same position on the white canvas
        white_background[y_min:y_max+1, x_min:x_max+1] = img[y1+y_min:y1+y_max+1, x1+x_min:x1+x_max+1].copy()

        image[y1:y2, x1:x2] = white_background.copy()
    return image


# custom torch transforms
class ResizeLongestSide:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        _, h, w = F.get_dimensions(img)        # works for PIL or tensor
        scale = self.size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        return F.resize(img, [new_h, new_w], interpolation=self.interpolation)
class PadToSquare:
    """Pad a PIL Image to a square by adding equal borders to the shorter side."""
    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size                      # PIL: (width, height)
        max_side = max(w, h)
        pad_w = max_side - w
        pad_h = max_side - h
        # left, top, right, bottom — split the difference, give the extra pixel to right/bottom
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )
        return F.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)

# Load transforms
def get_transforms(exp_params, transform):
    if exp_params['custom_transform'] is None:
        return transform
    elif exp_params['custom_transform'] == 'pad_resize_normalize':
        return T.Compose(
                [
                    PadToSquare(fill=0),
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=exp_params['norm_mu'], 
                                std=exp_params['norm_std']),
                ]
            )
    elif exp_params['custom_transform'] == 'pad_resize':
        return T.Compose(
                [
                    PadToSquare(fill=0),
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                ]
            )
    elif exp_params['custom_transform'] == 'pad_resize_pil':
        #is the version of the above transform for debugging -> i don't convert to tensor
        return T.Compose(
                [
                    PadToSquare(fill=0),
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                ]
            )
    elif exp_params['custom_transform'] == 'resize_normalize':
        return T.Compose(
                [
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=exp_params['norm_mu'], 
                                std=exp_params['norm_std']),
                ]
            )
    elif exp_params['custom_transform'] == 'resize':
        return T.Compose(
                [
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                ]
            )
    else:
        raise ValueError(f"Unknown custom_transform: {exp_params['custom_transform']}")

def get_augmentation_transform(exp_params):
    def get_single_augmentation_transform(t_modality):
        if t_modality is None:
            return None
        elif t_modality == 'random_crop_half':
            return T.Compose([
                T.RandomCrop(
                    int(exp_params['input_size'] / 2), 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ]) 
        elif t_modality == 'random_crop':
            return T.Compose([
                T.RandomCrop(
                    int(exp_params['input_size']), 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ]) 
        elif t_modality == 'grid':
            return 'grid'
        else:
            raise ValueError(f"Unknown augmentation_transform: {exp_params['apply_augmentation']}")
    if isinstance(exp_params['data_modality'], list):
        '''in this case i have to return a list of transforms
        for the grid sampling it returns grid because i cnanot pre-load a transform'''
        print("Multiple data modalities detected. Applying augmentation transforms based on modality:")
        list_of_transforms = []
        modality_to_transform = {
            'digit_full': None,
            'digit_crop': 'random_crop',
            'digit':'grid',
            'text_full': None,
            'text_crop': 'random_crop',
            'text':'grid',
            'X_crop' : 'random_crop',
            'X_full' : None,
            'X' : 'grid',
        }
        for t in exp_params['data_modality']:
            if t in modality_to_transform:
                list_of_transforms.append(get_single_augmentation_transform(modality_to_transform[t]))
            else:
                raise ValueError(f"Unknown data_modality: {t}")
        return list_of_transforms
    else:
        print("Single data modality detected. Applying augmentation transform:", exp_params['apply_augmentation'])
        return get_single_augmentation_transform(exp_params['apply_augmentation'])


# CHECK IMAGE PROPERTIES
def is_uniform_image(img_source, tol=0):
    """
    Check whether an image is a single uniform color (e.g. a blank/white
    imputed frame).

    tol = 0  -> strictly uniform (all pixels identical)
    tol > 0  -> uniform within tolerance (max - min <= tol per channel),
                catches near-white scans, mild JPEG noise, etc.

    Returns (is_uniform, is_white, fill_value).
    """
    arr = np.array(img_source)          # (H, W) for 'L', (H, W, C) for RGB/RGBA

    if arr.size == 0:
        return True, False, None

    if arr.ndim == 2:                   # grayscale
        lo, hi = int(arr.min()), int(arr.max())
        uniform = (hi - lo) <= tol
        value = int(round(arr.mean())) if uniform else None
        is_white = uniform and value >= (255 - tol)
    else:                               # multi-channel: check each channel
        lo = arr.min(axis=(0, 1))
        hi = arr.max(axis=(0, 1))
        uniform = bool(np.all((hi - lo) <= tol))
        value = tuple(int(round(v)) for v in arr.mean(axis=(0, 1))) if uniform else None
        is_white = uniform and bool(np.all(lo >= (255 - tol)))

    return bool(uniform), bool(is_white), value

def ink_mask(arr, threshold=128):
    """Boolean mask of ink pixels (darker than threshold)."""
    return arr < threshold

# --- blank / saturation ---
def is_blank(arr, threshold=128):
    """True if the image contains no ink pixels."""
    return bool((arr < threshold).sum() == 0)
def frac_pure_white(arr):
    """Fraction of pixels at value 255."""
    return float((arr == 255).mean())
def frac_pure_black(arr):
    """Fraction of pixels at value 0 (clipping/saturation)."""
    return float((arr == 0).mean())

# --- intensity ------
def ink_density(arr, threshold=128):
    ink_pixels = (arr < threshold).sum()
    #get the type of ink_density_binary
    #print(f"Type of ink_density_binary: {type(ink_density_binary)}")
    ink_density_binary = float(ink_pixels / arr.size)   # fraction of inked pixels
    return ink_density_binary
# --- ink-only intensity (independent of coverage) ---
def ink_intensity(arr, threshold=128):
    """(mean, std) over ink pixels only; (nan, nan) if no ink."""
    mask = arr < threshold
    if not mask.any():
        return float('nan'), float('nan')
    vals = arr[mask]
    return float(vals.mean()), float(vals.std())


# --- contrast / faded-scan detection ---

def dynamic_range(arr, low=1, high=99):
    """Spread between the low and high percentiles."""
    p_lo, p_hi = np.percentile(arr, [low, high])
    return float(p_hi - p_lo)

def otsu_threshold(arr):
    """Otsu's natural split point; nan if skimage missing or flat image.
    A fixed threshold of 128 silently breaks on faded or unevenly-lit scans. 
    Add the dynamic range and Otsu's natural split point — if Otsu drifts far from 128, 
    the image doesn't separate cleanly:"""
    return float(threshold_otsu(arr))


# --- digit geometry (bounding box of the ink) ---

def ink_geometry(arr, threshold=128):
    """
    Bounding box + extent + normalized centroid of the ink.
    Returns a dict; geometry fields are nan/0 when there is no ink.
    """
    H, W = arr.shape
    mask = arr < threshold
    ink_pixels = int(mask.sum())
    if ink_pixels == 0:
        return {'bbox_width': 0, 'bbox_height': 0, 'extent': float('nan'),
                'centroid_x': float('nan'), 'centroid_y': float('nan')}
    ys, xs = np.where(mask)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    return {
        'bbox_width':  bbox_w,
        'bbox_height': bbox_h,
        'extent':      float(ink_pixels / (bbox_w * bbox_h)),
        'centroid_x':  float(xs.mean() / W),
        'centroid_y':  float(ys.mean() / H),
    }


# --- blur / sharpness ---

def sharpness(arr):
    """Variance of the Laplacian; higher = sharper. 
     Laplacian variance is the standard sharpness proxy — low values mean blurry:"""
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


# --- connected components (speckle / noise) ---

def n_components(arr, threshold=128, min_size=0):
    """
    Count connected ink blobs. Optionally drop components smaller than
    min_size pixels.
    """
    labels, n = ndimage.label(arr < threshold)
    if min_size > 0 and n > 0:
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, n + 1))
        n = int((sizes >= min_size).sum())
    return int(n)
   