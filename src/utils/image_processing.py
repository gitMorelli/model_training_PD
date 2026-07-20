import cv2
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import os
from skimage.filters import threshold_otsu 
from scipy import ndimage
from PIL import ImageFilter
import random
import math
import torch
import math
import random

import cv2
import numpy as np
from PIL import Image

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


# ----------------- custom torch transforms ---------------------
class ResizeLongestSide:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        _, h, w = TF.get_dimensions(img)        # works for PIL or tensor
        scale = self.size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        return TF.resize(img, [new_h, new_w], interpolation=self.interpolation)
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
        return TF.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)
class PadOrCropToSize:
    """Force an image to size x size by center-padding and/or center-cropping.
    Handles each axis independently: an image can be padded on one axis
    and cropped on the other."""

    def __init__(self, size, fill=0, padding_mode="constant"):
        self.size = (size, size) if isinstance(size, int) else size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        th, tw = self.size
        w, h = img.size          # PIL is (w, h)

        # --- 1. Pad any axis that is too small (no-op if already >= target)
        pad_w = max(tw - w, 0)
        pad_h = max(th - h, 0)
        if pad_w or pad_h:
            left = pad_w // 2
            top = pad_h // 2
            img = TF.pad(
                img,
                [left, top, pad_w - left, pad_h - top],   # l, t, r, b
                fill=self.fill,
                padding_mode=self.padding_mode,
            )

        # --- 2. Crop any axis that is too large (no-op if already == target)
        return TF.center_crop(img, [th, tw])

# synthetic transformations
ALL_SYNTHETIC_TRANSFORMS = [
    "original",
    "progressive_thickening",
    "progressive_thinning",
    "progressive_slant",
    "progressive_size_drift",
    "progressive_baseline_wave",
    "progressive_tremor",
    "progressive_ink_density",
    'progressive_size_drift_x',
    'progressive_size_drift_y',
]

class RandomMorphology:
    """Erosion (min filter) thickens black ink; dilation (max filter) thins it.
    MinFilter/MaxFilter only accept odd sizes (3, 5, 7…)"""
    def __init__(self, p=0.5, kernel_sizes=(3,), mode="erode"):
        self.p = p
        self.kernel_sizes = kernel_sizes
        self.mode = mode

    def __call__(self, img):
        if random.random() > self.p:
            return img
        k = random.choice(self.kernel_sizes)
        mode = self.mode
        if mode == "random":
            mode = random.choice(["erode", "dilate"])
        f = ImageFilter.MinFilter(k) if mode == "erode" else ImageFilter.MaxFilter(k)
        return img.filter(f)

# Default source of per-call randomness (phase / tremor field). Inject a seeded
# np.random.Generator into the stochastic transforms for reproducibility.
_RNG = np.random.default_rng()

# Cache of identity coordinate grids keyed by (H, W). remap needs contiguous
# float32 map_x / map_y the size of the output; the base grids only depend on
# the image size, so we build them once.
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _identity_grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    key = (H, W)
    g = _GRID_CACHE.get(key)
    if g is None:
        xs, ys = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        g = (np.ascontiguousarray(xs), np.ascontiguousarray(ys))
        _GRID_CACHE[key] = g
    return g


def _affine_inv(ink: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply an output->input (inverse) 2x3 affine map with bilinear sampling."""
    H, W = ink.shape[:2]
    return cv2.warpAffine(
        ink, M, (W, H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

class StrokeWeight:
    """t -> stroke gets thicker (w>0) or thinner (w<0). Fractional via blending.

    max_pool2d(stride 1) is grayscale dilation with a flat square element;
    -max_pool2d(-x) is erosion. Fractional growth is a blend toward the pooled
    image, exactly as in the original.
    """

    def __init__(self, rate: float = 1.0, max_px: float = 2.5):
        self.rate, self.max_px = rate, max_px

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        w = self.rate * t * self.max_px
        if abs(w) < 1e-3:
            return ink
        k = 2 * int(math.ceil(abs(w))) + 1
        a = abs(w) / (k // 2)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        pooled = cv2.dilate(ink, se) if w > 0 else cv2.erode(ink, se)
        return (1.0 - a) * ink + a * pooled

class Slant:
    """t -> increasing shear. Positive = leaning right.

    grid_sample shear x' = x + tan(ang)*y in the normalised box; in pixels the
    coefficient picks up a W/H factor, applied about the vertical centre.
    """

    def __init__(self, rate: float = 1.0, max_deg: float = 30.0):
        self.rate, self.max_deg = rate, max_deg

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        ang = math.radians(self.rate * t * self.max_deg)
        H, W = ink.shape[:2]
        cy = (H - 1) / 2.0
        s = math.tan(ang) * (W / H)          # aspect-corrected shear
        M = np.array([[1.0, s, -s * cy],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        return _affine_inv(ink, M)

class SizeDrift:
    """t -> handwriting grows or shrinks; can be anisotropic (taller/narrower).

    Inverse (output->input) scale about the centre: sample = centre + (px-centre)/s.
    """

    def __init__(self, rate_x: float = 0.0, rate_y: float = 0.0, max_frac: float = 2):
        self.rx, self.ry, self.m = rate_x, rate_y, max_frac

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        sx = 1.0 + self.rx * t * self.m
        sy = 1.0 + self.ry * t * self.m
        # Guard against a zero / negative-collapse blow-up (the original 1/sx
        # would also misbehave here); keep the sign, clamp the magnitude.
        sx = math.copysign(max(abs(sx), 1e-2), sx or 1.0)
        sy = math.copysign(max(abs(sy), 1e-2), sy or 1.0)
        H, W = ink.shape[:2]
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        M = np.array([[1.0 / sx, 0.0, cx * (1.0 - 1.0 / sx)],
                      [0.0, 1.0 / sy, cy * (1.0 - 1.0 / sy)]], dtype=np.float32)
        return _affine_inv(ink, M)

class BaselineWave:
    """t -> the writing line stops being straight; low-frequency vertical wobble."""

    def __init__(self, rate: float = 1.0, max_amp: float = 0.1,
                 cycles: float = 1.5, rng=None):
        self.rate, self.max_amp, self.cycles = rate, max_amp, cycles
        self.rng = rng if rng is not None else _RNG

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        amp = self.rate * t * self.max_amp
        if amp < 1e-4:
            return ink
        H, W = ink.shape[:2]
        xs, ys = _identity_grid(H, W)
        phase = self.rng.uniform(0.0, 2 * math.pi)
        u = np.linspace(-1.0, 1.0, W, dtype=np.float32)           # wave coord
        # normalised amp -> pixels via H/2
        dy = (amp * (H / 2.0)) * np.sin(self.cycles * math.pi * u + phase)
        map_y = np.ascontiguousarray(ys + dy[None, :].astype(np.float32))
        return cv2.remap(ink, xs, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

class Tremor:
    """t -> shaky hand: smooth random displacement field, amplitude grows with t.

    Two box filters reproduce the original's two avg_pool2d passes (avg pool of
    stride 1 == mean/box filter).
    """

    def __init__(self, rate: float = 1.0, max_amp: float = 0.1,
                 smooth: int = 8, rng=None):
        self.rate, self.max_amp, self.smooth = rate, max_amp, smooth
        self.rng = rng if rng is not None else _RNG

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        amp = self.rate * t * self.max_amp
        if amp < 1e-4:
            return ink
        H, W = ink.shape[:2]
        xs, ys = _identity_grid(H, W)
        k = self.smooth | 1
        d = self.rng.standard_normal((H, W, 2)).astype(np.float32)
        d = cv2.boxFilter(d, -1, (k, k), normalize=True)
        d = cv2.boxFilter(d, -1, (k, k), normalize=True)         # ~ gaussian
        m = float(np.abs(d).max())
        d *= amp / (m if m > 1e-6 else 1e-6)
        map_x = np.ascontiguousarray(xs + d[:, :, 0] * (W / 2.0))
        map_y = np.ascontiguousarray(ys + d[:, :, 1] * (H / 2.0))
        return cv2.remap(ink, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

class InkDensity:
    """t -> pen runs dry (fading, patchy) or presses harder (darker, blotchy)."""

    def __init__(self, rate: float = -1.0, max_gamma: float = 8):
        self.rate, self.max_gamma = rate, max_gamma

    def __call__(self, ink: np.ndarray, t: float) -> np.ndarray:
        g = 1.0 + self.rate * t * self.max_gamma     # >1 fades, <1 darkens
        g = max(g, 0.02)
        return np.clip(ink, 0.0, 1.0) ** np.float32(g)

def to_ink(pil_img: Image.Image) -> np.ndarray:
    """PIL (0-255, black-on-white) -> float32 array in [0,1], ink=1.

    Shape is (H,W) for 'L' or (H,W,C) for multi-channel; every transform is
    channel-agnostic, so both work.
    """
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return 1.0 - arr

def to_img(ink: np.ndarray) -> Image.Image:
    """float32 ink -> PIL (0-255, black-on-white)."""
    arr = np.clip(1.0 - ink, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr))

class SyntheticTransform:
    def __init__(self, exp_params, subject_id, train_df, persona_seed,
                 n_steps=10, jitter=0.15):
        synthetic_transform_names = exp_params['synthetic']
        class_value = train_df.loc[
            train_df['unique_id'] == subject_id, 'synth_label'
        ].values[0]
        selected_transform = synthetic_transform_names[class_value]

        rng = random.Random(persona_seed)
        u = lambda: rng.uniform(-1, 1)          # drift direction+rate per persona
        nrng = np.random.default_rng(persona_seed)   # per-persona field/phase RNG
        self.n_steps, self.jitter = n_steps, jitter

        if selected_transform == 'original':
            self.transform = None
        elif selected_transform == 'progressive_thickening':
            self.transform = StrokeWeight(rate=abs(u()))
        elif selected_transform == 'progressive_thinning':
            self.transform = StrokeWeight(rate=-abs(u()))
        elif selected_transform == 'progressive_slant':
            self.transform = Slant(rate=u())
        elif selected_transform == 'progressive_size_drift':
            self.transform = SizeDrift(rate_x=u(), rate_y=u())
        elif selected_transform == 'progressive_size_drift_x':
            self.transform = SizeDrift(rate_x=u(), rate_y=u() * 0.1)
        elif selected_transform == 'progressive_size_drift_y':
            self.transform = SizeDrift(rate_x=u() * 0.1, rate_y=u())
        elif selected_transform == 'progressive_baseline_wave':
            self.transform = BaselineWave(rate=abs(u()), rng=nrng)
        elif selected_transform == 'progressive_tremor':
            self.transform = Tremor(rate=abs(u()), rng=nrng)
        elif selected_transform == 'progressive_ink_density':
            self.transform = InkDensity(rate=u())
        else:
            raise ValueError(f"unknown transform: {selected_transform!r}")

    def __call__(self, img: Image.Image, step: int) -> Image.Image:
        if self.transform is None:
            return img
        t = step / max(self.n_steps - 1, 1)
        ink = to_ink(img)
        tt = max(0.0, t * (1.0 + random.uniform(-self.jitter, self.jitter)))
        ink = self.transform(ink, tt)
        return to_img(ink)

#---------- Load transforms ---------------
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
            t=None
        elif t_modality == 'random_crop_half':
            t = T.Compose([
                T.RandomCrop(
                    int(exp_params['input_size'] / 2), 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ]) 
        elif t_modality == 'random_crop':
            t = T.Compose([
                T.RandomCrop(
                    int(exp_params['input_size']), 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ]) 
        elif t_modality == 'grid':
            #pad to 64x64 
            t = T.Compose([
                PadOrCropToSize(64, fill=(255,255,255)) #pad shortersides and then centerCrop
            ])
        else:
            raise ValueError(f"Unknown augmentation_transform: {exp_params['apply_augmentation']}")
        return (t_modality,t)
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

def get_mu_std(exp_params,verbose=False):
    #if it is a list of numbers keep it as is, if it is a string load the mu and std from the file
    if isinstance(exp_params['norm_mu'], (list, tuple)) and isinstance(exp_params['norm_std'], (list, tuple)):
        mu, std = exp_params['norm_mu'], exp_params['norm_std']
    elif isinstance(exp_params['norm_mu'], str) or isinstance(exp_params['norm_std'], str):
        if exp_params['norm_mu']=='mnist':
            mu = (0.1307,0.1307,0.1307)
            std = (0.3081,0.3081,0.3081)
        elif exp_params['norm_mu']=='imagenet':
            mu = (0.485,0.456,0.406)
            std = (0.229,0.224,0.225)
        elif exp_params['norm_mu']=='handedness':
            mu = [0.06040578708052635, 0.06040578708052635, 0.06040578708052635]
            std = [0.23823712766170502, 0.23823712766170502, 0.23823712766170502]
    if verbose:
        print(f"Using normalization mean: {mu} and std: {std}")
    return mu, std


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
   