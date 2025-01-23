import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import Pool, cpu_count
import re
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from lpips import LPIPS
import logging
import nibabel as nib
import kagglehub
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.nn.utils import spectral_norm
from torch_optimizer import Lookahead, RAdam
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, Dice, JaccardIndex
from torch.distributions import Normal
import random
from typing import Tuple, List
from torch.nn import functional as F
from math import sqrt
from timm.models import resnet
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn.functional import interpolate
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch.optim.lr_scheduler import CyclicLR
from itertools import cycle, islice
from collections import deque
import kornia.augmentation as K
import kornia.geometry.transform as T



class Config:

    BATCH_SIZE = 1  
    IMAGE_SIZE = 254
    LOW_RES_SIZE = IMAGE_SIZE // 4
    CHANNELS = 1
    EPOCHS = 300
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True
    
    NUM_RESIDUAL_BLOCKS = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    LAMBDA_ADV = 0.001
    LAMBDA_PERC = 0.1
    LAMBDA_CONTENT = 1.0
    LAMBDA_SSIM = 0.1
    LAMBDA_FEATURE = 0.1
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.2
    WARMUP_STEPS = 200
    
    # Model Parameters
    NUM_HEADS = 8
    EMBED_DIM = 64  # Base embedding dim, can be modified
    
    # Data loading
    NUM_WORKERS = 12
    PIN_MEMORY = True
    
    # Augmentation parameters
    AFFINE_DEGREES = 15
    AFFINE_TRANSLATE = (0.1, 0.1, 0.1) 
    AFFINE_SCALE = (0.8, 1.2)

    # Patch size for 3D data
    PATCH_SIZE = 64
    PATCH_STRIDE = PATCH_SIZE // 2 # Controls patch overlapping
    
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(BASE_DIR, "brats_dataset")
    TRAIN_HR_FOLDER = os.path.join(DATASET_FOLDER, "train_HR")
    TRAIN_LR_FOLDER = os.path.join(DATASET_FOLDER, "train_LR")
    VAL_HR_FOLDER = os.path.join(DATASET_FOLDER, "val_HR")
    VAL_LR_FOLDER = os.path.join(DATASET_FOLDER, "val_LR")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    GENERATED_DIR = os.path.join(BASE_DIR, "generated_medical")
    
    # Distributed Training
    DISTRIBUTED = False
    LOCAL_RANK = None
    WORLD_SIZE = None
    
    # Loss weights
    ADV_WEIGHT = 0.1
    CONTENT_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT = 0.1
    SSIM_WEIGHT = 0.1
    SEG_WEIGHT = 0.1 

    # Perceptual loss weights
    PERC_LAYER_WEIGHTS = [1, 0.5, 0.25, 0.125]  

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15  
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Cross validation
    NUM_FOLDS = 5

    # Self Supervised learning parameters
    SELF_SUPERVISED_EPOCHS = 50
    SELF_SUPERVISED_BATCH_SIZE = 16

    # Data Preprocessing
    DATA_NORM_MEAN = 0.5
    DATA_NORM_STD = 0.5

    # Activation Analysis
    ACTIVATION_LAYERS = [1, 7]  # Layers to visualize in the generator
    ACTIVATION_SAMPLE_COUNT = 4  # Number of images to use for activation visualization

    # Cyclical LR
    CYCLE_LR_BASE_LR = 1e-5
    CYCLE_LR_MAX_LR = 1e-4
    CYCLE_LR_STEP_SIZE_UP = 1000
    CYCLE_LR_MODE = "triangular2"

    # Progressive Growing
    PROGRESSIVE_GROW_STEPS = 5  # Number of steps at each image size.

    # Data Subsetting
    SUBSET_SIZE = 0.01

    # Quantile Clipping
    QUANTILE_CLIP_MIN = 0.01
    QUANTILE_CLIP_MAX = 0.99

    # Validation subset size
    VALIDATION_SUBSET_SIZE = 0.01
    
    @staticmethod
    def set_dataset_folder(path):
        Config.DATASET_FOLDER = path
        Config.TRAIN_HR_FOLDER = os.path.join(path, "train_HR")
        Config.TRAIN_LR_FOLDER = os.path.join(path, "train_LR")
        Config.VAL_HR_FOLDER = os.path.join(path, "val_HR")
        Config.VAL_LR_FOLDER = os.path.join(path, "val_LR")

def is_main_process():
    return not Config.DISTRIBUTED or Config.LOCAL_RANK == 0

def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    log_filename = os.path.join(Config.LOG_DIR, 'training.log') if is_main_process() else None
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    if is_main_process():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        Config.DISTRIBUTED = True
        Config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(Config.LOCAL_RANK)
        init_process_group(backend="nccl")
        Config.WORLD_SIZE = torch.distributed.get_world_size()
        logging.info(f"Distributed training enabled. Rank: {Config.LOCAL_RANK}, World size: {Config.WORLD_SIZE}")
    else:
        logging.info("Distributed training not enabled")

def cleanup_distributed():
    if Config.DISTRIBUTED:
        destroy_process_group()

def download_brats_dataset():
    if is_main_process():
        logging.info("Downloading BraTS dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("shakilrana/brats-2023-adult-glioma")
            logging.info(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise
    else:
        logging.info("Waiting for main process to download dataset...")
        torch.distributed.barrier()
        return Config.BASE_DIR + "/brats-2023-adult-glioma"

def resize_with_padding_numpy_3d(volume, target_size):
    """Resizes and pads a 3D volume with numpy only."""    
    depth, height, width = volume.shape
    target_depth, target_height, target_width = target_size
    
    target_ratio_xy = target_width / target_height
    target_ratio_xz = target_width / target_depth
    target_ratio_yz = target_height / target_depth
    
    img_ratio_xy = width / height
    img_ratio_xz = width / depth
    img_ratio_yz = height / depth

    if img_ratio_xy > target_ratio_xy:
         new_width_xy = target_width
         new_height_xy = int(new_width_xy / img_ratio_xy)
    else:
        new_height_xy = target_height
        new_width_xy = int(new_height_xy * img_ratio_xy)
    
    if img_ratio_xz > target_ratio_xz:
        new_width_xz = target_width
        new_depth_xz = int(new_width_xz / img_ratio_xz)
    else:
        new_depth_xz = target_depth
        new_width_xz = int(new_depth_xz * img_ratio_xz)

    if img_ratio_yz > target_ratio_yz:
        new_height_yz = target_height
        new_depth_yz = int(new_height_yz / img_ratio_yz)
    else:
        new_depth_yz = target_depth
        new_height_yz = int(new_depth_yz * img_ratio_yz)

    new_width = int(min(new_width_xy, new_width_xz))
    new_height = int(min(new_height_xy, new_height_yz))
    new_depth = int(min(new_depth_xz, new_depth_yz))
   
    resized_volume = np.array(Image.fromarray(volume.transpose(1, 2, 0), mode = "F").resize((new_width, new_height, new_depth), resample=Image.BICUBIC))

    padded_volume = np.zeros(target_size, dtype=resized_volume.dtype)

    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    front = (target_depth - new_depth) // 2

    padded_volume[front:front+new_depth, top:top+new_height, left:left+new_width] = resized_volume.transpose(2,0,1)
    return padded_volume

def apply_affine_transform_3d(volume, affine_matrix):
    """Applies an affine transformation to a 3D volume."""
    transformed_volume = affine_transform(volume, affine_matrix)
    return transformed_volume

def process_file_chunk(args):
    (file_path, patch_size, patch_stride, use_segmentation, hr_folder, lr_folder, val_folder, subset, validation_subset) = args
    try:
        logging.info(f"Preprocessing: Started processing: {file_path}")
        img = nib.load(file_path)
        img_data = np.asanyarray(img.get_fdata())
        
        # Normalize data to [0, 1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Extract modality from filename
        modality = os.path.basename(file_path).split("_")[-1].split(".")[0]

        # Ensure 3D array and take all channels if present
        if img_data.ndim == 4:
            num_channels = img_data.shape[3]
        else:
            num_channels = 1
            img_data = np.expand_dims(img_data, axis=3)
        
        num_patches_processed = 0
        
        for channel_idx in range(num_channels):
            volume = img_data[:, :, :, channel_idx]

            # Quantile Clipping
            lower_quantile = np.quantile(volume, Config.QUANTILE_CLIP_MIN)
            upper_quantile = np.quantile(volume, Config.QUANTILE_CLIP_MAX)
            volume = np.clip(volume, lower_quantile, upper_quantile)
           
            # Affine Transformation before resizing
            angle_x = np.random.uniform(-Config.AFFINE_DEGREES, Config.AFFINE_DEGREES)
            angle_y = np.random.uniform(-Config.AFFINE_DEGREES, Config.AFFINE_DEGREES)
            angle_z = np.random.uniform(-Config.AFFINE_DEGREES, Config.AFFINE_DEGREES)

            scale = np.random.uniform(Config.AFFINE_SCALE[0], Config.AFFINE_SCALE[1])

            translate = (
                  np.random.uniform(-Config.AFFINE_TRANSLATE[0], Config.AFFINE_TRANSLATE[0]),
                  np.random.uniform(-Config.AFFINE_TRANSLATE[1], Config.AFFINE_TRANSLATE[1]),
                  np.random.uniform(-Config.AFFINE_TRANSLATE[2], Config.AFFINE_TRANSLATE[2])
              )

            # Create transformation matrix
            center = np.array(volume.shape) / 2.0
            
            # Create a rotation matrix for each axis
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(np.deg2rad(angle_x)), -np.sin(np.deg2rad(angle_x))],
                [0, np.sin(np.deg2rad(angle_x)), np.cos(np.deg2rad(angle_x))]
            ])

            Ry = np.array([
                [np.cos(np.deg2rad(angle_y)), 0, np.sin(np.deg2rad(angle_y))],
                [0, 1, 0],
                [-np.sin(np.deg2rad(angle_y)), 0, np.cos(np.deg2rad(angle_y))]
            ])

            Rz = np.array([
                [np.cos(np.deg2rad(angle_z)), -np.sin(np.deg2rad(angle_z)), 0],
                [np.sin(np.deg2rad(angle_z)), np.cos(np.deg2rad(angle_z)), 0],
                [0, 0, 1]
            ])

            # Apply scaling matrix
            S = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, scale]
            ])
            
            # Combine rotation matrices
            R = np.dot(Rz, np.dot(Ry, Rx))

            # Create transform matrix
            transform_matrix = np.eye(4) # 4x4 Identity matrix
            transform_matrix[:3, :3] = np.dot(S,R) # Rotation and scaling components
            transform_matrix[:3, 3] = translate

            transform_matrix[0, 3] += -center[0] * transform_matrix[0, 0] - center[1] * transform_matrix[0, 1] - center[2] * transform_matrix[0, 2] + center[0]
            transform_matrix[1, 3] += -center[0] * transform_matrix[1, 0] - center[1] * transform_matrix[1, 1] - center[2] * transform_matrix[1, 2] + center[1]
            transform_matrix[2, 3] += -center[0] * transform_matrix[2, 0] - center[1] * transform_matrix[2, 1] - center[2] * transform_matrix[2, 2] + center[2]
            
            volume = apply_affine_transform_3d(volume, transform_matrix[:3, :3].T)

             # Resize and pad the transformed volume
            volume = torch.tensor(volume, dtype = torch.float32).unsqueeze(0).unsqueeze(0) # Convert to torch tensor and add batch and channel dimensions
            try:
                 volume = F.interpolate(volume, size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.IMAGE_SIZE), mode="trilinear", align_corners=False).squeeze(0).squeeze(0).numpy()
            except Exception as e:
                 logging.error(f"Error when resampling volume to size ({Config.IMAGE_SIZE},{Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}) and shape {volume.shape}, Error: {e}")
                 raise

            # Generate Patches 
            depth, height, width = volume.shape
            
            #Generate indices
            z_indices = list(range(0, depth - patch_size + 1, patch_stride))
            y_indices = list(range(0, height - patch_size + 1, patch_stride))
            x_indices = list(range(0, width - patch_size + 1, patch_stride))

            # Generate all patches at once
            for z in z_indices:
               for y in y_indices:
                  for x in x_indices:
                        hr_patch = volume[z:z + patch_size, y:y + patch_size, x:x + patch_size]

                        # Generate low-resolution patch
                        lr_patch = torch.tensor(hr_patch, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
                        lr_patch = F.interpolate(lr_patch, size = (patch_size // 4, patch_size // 4, patch_size // 4), mode = "trilinear", align_corners=False)
                        lr_patch = lr_patch.squeeze(0).squeeze(0).numpy()
                        
                        # Convert to torch tensors
                        hr_tensor = torch.tensor(hr_patch, dtype=torch.float32).unsqueeze(0) # Add channel dimension
                        lr_tensor = torch.tensor(lr_patch, dtype=torch.float32).unsqueeze(0) # Add channel dimension

                        # Save tensors
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        patch_name = f'{base_name}_{channel_idx}_{z}_{y}_{x}'
                        
                        is_val = False
                        if val_folder is not None:
                            is_val = True
                        
                        if is_val:
                            hr_file_path = os.path.join(hr_folder, f'{patch_name}_HR.pt')
                            lr_file_path = os.path.join(lr_folder, f'{patch_name}_LR.pt')
                            torch.save(hr_tensor, hr_file_path)
                            torch.save(lr_tensor, lr_file_path)
                            # Save segmentation masks for all modalities
                            if use_segmentation:
                                    seg_volume = volume > 0.3  # Threshold
                                    seg_patch = seg_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size]
                                    seg_tensor = torch.tensor(seg_patch, dtype=torch.float32).unsqueeze(0)
                                    seg_file_path = os.path.join(hr_folder, f'{patch_name}_SEG.pt')
                                    torch.save(seg_tensor, seg_file_path)
                        else:
                            hr_file_path = os.path.join(hr_folder, f'{patch_name}_HR.pt')
                            lr_file_path = os.path.join(lr_folder, f'{patch_name}_LR.pt')
                            torch.save(hr_tensor, hr_file_path)
                            torch.save(lr_tensor, lr_file_path)
                             # Save segmentation masks for all modalities
                            if use_segmentation:
                                seg_volume = volume > 0.3 # Threshold
                                seg_patch = seg_volume[z:z + patch_size, y:y + patch_size, x:x + patch_size]
                                seg_tensor = torch.tensor(seg_patch, dtype=torch.float32).unsqueeze(0)
                                seg_file_path = os.path.join(hr_folder, f'{patch_name}_SEG.pt')
                                torch.save(seg_tensor, seg_file_path)
                        num_patches_processed += 1
                    

        logging.info(f"Preprocessing: Finished processing: {file_path}. Processed {num_patches_processed} patches.")
        return num_patches_processed
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return 0

def preprocess_dataset(use_segmentation=False, image_size=None, validation=False, subset = 1.0, validation_subset = 1.0):
    hr_folder = Config.TRAIN_HR_FOLDER
    lr_folder = Config.TRAIN_LR_FOLDER
    val_folder = None
    subset = subset
    validation_subset = validation_subset

    if validation:
        hr_folder = Config.VAL_HR_FOLDER
        lr_folder = Config.VAL_LR_FOLDER
        val_folder = True

    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)

    num_patches = 0

    logging.info(f"Preprocessing: Scanning dataset folder: {Config.DATASET_FOLDER}")

    # Use all available CPU cores for processing.
    num_processes = cpu_count()
    logging.info(f"Preprocessing: Using {num_processes} processes")
    with Pool(processes = num_processes) as pool:
        tasks = []
        try:
            for dirpath, _, filenames in os.walk(Config.DATASET_FOLDER):
                logging.info(f"Preprocessing: Files in current folder: {dirpath} , Files found: {filenames}") #Added log for every folder found.
                #Only look in training data path
                if "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" not in dirpath:
                   continue

                #Gather all files that match the pattern
                file_paths = [os.path.join(dirpath, file_name) for file_name in filenames if re.match(r'^\d+_brain_(flair|t1|t1ce|t2).nii', file_name)]
                
                for file_path in file_paths: #Iterate throught the list
                    if not validation:
                       if np.random.rand() < subset:
                           tasks.append((file_path, Config.PATCH_SIZE, Config.PATCH_STRIDE, use_segmentation, hr_folder, lr_folder, val_folder, subset, validation_subset)) # Appends single files
                    else:
                        if np.random.rand() < validation_subset:
                            tasks.append((file_path, Config.PATCH_SIZE, Config.PATCH_STRIDE, use_segmentation, hr_folder, lr_folder, val_folder, subset, validation_subset))  # Appends single files
            results = pool.imap_unordered(process_file_chunk, tasks) # Changed to imap

            num_patches = sum(results)

        except Exception as e:
            logging.error(f"Failed to process dataset: {e}")
            raise
            
    logging.info(f"Preprocessing: Created {num_patches} total image patches.")
    return num_patches


class BraTSDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=True, norm=True, subset = 1.0, progressive = False, image_size = None, use_segmentation = True):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.norm = norm
        self.subset = subset
        self.progressive = progressive
        self.image_size = image_size
        self.use_segmentation = use_segmentation  

        hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('_HR.pt')])
        lr_files = sorted([f for f in os.listdir(lr_folder) if f.endswith('_LR.pt')])
        
        self.image_pairs = [(
            os.path.join(hr_folder, hr_file),
            os.path.join(lr_folder, hr_file.replace('_HR.pt', '_LR.pt'))
        ) for hr_file in hr_files if hr_file.replace('_HR.pt', '_LR.pt') in lr_files]
        
        if self.subset < 1.0:
            num_subset = int(len(self.image_pairs) * self.subset)
            self.image_pairs = self.image_pairs[:num_subset]
        
        if transform:
            self.transform = transforms.Compose([
        RandomHorizontalFlip3D(p=0.5),
        RandomVerticalFlip3D(p=0.5),
        RandomDepthFlip3D(p=0.5),
        RandomRotation3D(degrees=Config.AFFINE_DEGREES, p=0.3),  # Reduced probability
        RandomAffine3D(
            degrees=Config.AFFINE_DEGREES,
            translate=Config.AFFINE_TRANSLATE,
            scale=Config.AFFINE_SCALE,
            p=0.3  # Reduced probability
        ),
        RandomElasticDeformation3D(alpha=10, sigma=3, p=0.3),  # Reduced probability
        RandomApply(GaussianBlur3D(kernel_size=3, sigma=(0.1, 2)), p=0.2),
        RandomApply(AddGaussianNoise3D(0, 0.1), p=0.2),
        RandomApply(Cutout3D(length=16), p=0.2)
        ])
        else:
            self.transform = None
        
        logging.info(f"Found {len(self.image_pairs)} valid image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.image_pairs[idx]
        
        try:
            hr_tensor = torch.load(hr_path, weights_only=False).float()
            lr_tensor = torch.load(lr_path, weights_only=False).float()
            
            if self.norm:
                hr_tensor = (hr_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
                lr_tensor = (lr_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
            
            if hr_tensor.ndim != 4 or lr_tensor.ndim != 4:
                hr_tensor = hr_tensor.unsqueeze(0) if hr_tensor.ndim == 3 else hr_tensor
                lr_tensor = lr_tensor.unsqueeze(0) if lr_tensor.ndim == 3 else lr_tensor
            
            hr_tensor = apply_transform(hr_tensor, self.transform)
            lr_tensor = apply_transform(lr_tensor, self.transform)
            
            #Remove the channel dimension if it has 2 channels
            if lr_tensor.shape[0] == 2:
                  lr_tensor = lr_tensor[0].unsqueeze(0)
                
            if self.use_segmentation:
                seg_path = hr_path.replace('_HR.pt', '_SEG.pt')
                if os.path.exists(seg_path):
                    seg_tensor = torch.load(seg_path, weights_only=False).float()
                    if self.norm:
                        seg_tensor = (seg_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
                    seg_tensor = apply_transform(seg_tensor, self.transform)
                else:
                    # Generate a placeholder mask (all zeros)
                    seg_tensor = torch.zeros_like(hr_tensor)
                    logging.warning(f"Segmentation mask not found at: {seg_path}. Using a placeholder mask.")
                
                return lr_tensor, hr_tensor, seg_tensor
            else:
                return lr_tensor, hr_tensor
            
        except Exception as e:
            logging.error(f"Error loading images at index {idx}: {e}")
            raise
        
def apply_transform(tensor, transform):
    """Apply transforms with proper dimension handling."""
    if transform is None:
        return tensor
    
    if not isinstance(tensor, torch.Tensor):
        logging.error(f"Invalid type for transformation: {type(tensor)}")
        return tensor

    # Store original shape and dimensions
    orig_shape = tensor.shape
    orig_ndim = tensor.ndim

    try:
        # Ensure 5D tensor [B,C,D,H,W]
        if tensor.ndim == 3:  # [D,H,W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:  # [C,D,H,W]
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:
            raise ValueError(f"Invalid tensor dimensions: {tensor.shape}")

        # Apply transform
        transformed = transform(tensor)

        # Validate output dimensions
        if transformed.ndim != 5:
            raise ValueError(f"Transform output has wrong dimensions: {transformed.shape}")

        if transformed.shape[2:] != tensor.shape[2:]:
            transformed = F.interpolate(
                transformed,
                size=tensor.shape[2:],  # [D,H,W]
                mode='trilinear',
                align_corners=True
            )

        # Restore original dimensions
        if orig_ndim == 3:
            transformed = transformed.squeeze(0).squeeze(0)
        elif orig_ndim == 4:
            transformed = transformed.squeeze(0)

        return transformed

    except Exception as e:
        logging.error(f"Transform error: {str(e)}")
        return tensor


class RandomApply(torch.nn.Module):
    """Applies a transformation with a given probability."""
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            return self.transform(x)
        return x
    
class GaussianBlur3D(torch.nn.Module):
    """Applies gaussian blur to a 3D tensor."""
    def __init__(self, kernel_size, sigma=(0.1, 2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.rng = np.random.default_rng()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
            
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2)
        
    def forward(self, x):
        sigma = self.rng.uniform(self.sigma[0], self.sigma[1])
        
        # Generate Gaussian kernel
        k_x = self._get_gaussian_kernel(self.kernel_size[0], sigma).to(x.device)
        k_y = self._get_gaussian_kernel(self.kernel_size[1], sigma).to(x.device)
        k_z = self._get_gaussian_kernel(self.kernel_size[2], sigma).to(x.device)


        # Prepare the kernel for convolution
        kernel = k_x.reshape(1, 1, -1, 1, 1) * k_y.reshape(1, 1, 1, -1, 1) * k_z.reshape(1, 1, 1, 1, -1)
        kernel = kernel / kernel.sum()
        
        # Apply convolution
        b,c,d,h,w = x.shape
        output = []
        for i in range(c):
            output.append(F.conv3d(
                x[:,i:i+1],
                kernel,
                padding=self.padding,
                groups=1  #applying on each channel separately
            ))
        
        return torch.cat(output, dim = 1) # concatanete the channels
    
    def _get_gaussian_kernel(self, kernel_size, sigma):
        """Generate 1D Gaussian kernel."""
        x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1)
        return torch.exp(-x**2 / (2 * sigma**2)).float()
    
class AddGaussianNoise3D(torch.nn.Module):
    """Applies random gaussian noise to a 3D tensor."""
    def __init__(self, mean=0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng()

    def forward(self, x):
        std = self.rng.uniform(0, self.std)
        noise = torch.randn_like(x) * std + self.mean
        return x + noise
    
class Cutout3D(torch.nn.Module):
    """Applies Cutout augmentation to a 3D tensor."""
    def __init__(self, length=16):
        super().__init__()
        self.length = length
        self.rng = np.random.default_rng()

    def forward(self, img):
        d, h, w = img.shape[-3:]
        mask = torch.ones_like(img)
        
        z = self.rng.integers(0, max(1, d - min(self.length, d)))
        y = self.rng.integers(0, max(1, h - min(self.length,h)))
        x = self.rng.integers(0, max(1, w - min(self.length,w)))
        
        cutout_length_z = min(self.length, d)
        cutout_length_y = min(self.length, h)
        cutout_length_x = min(self.length, w)

        mask[..., z:z+cutout_length_z, y:y+cutout_length_y, x:x+cutout_length_x] = 0
        return img * mask
    
class RandomHorizontalFlip3D(torch.nn.Module):
    """Applies Random horizontal flip to a 3D tensor."""
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            return torch.flip(x, dims=[-1]) # Horizontal flip on the last dimension (width)
        return x

class RandomVerticalFlip3D(torch.nn.Module):
    """Applies Random vertical flip to a 3D tensor."""
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            return torch.flip(x, dims=[-2]) # Vertical flip on the second to last dimension (height)
        return x
    
class RandomDepthFlip3D(torch.nn.Module):
    """Applies Random depth flip to a 3D tensor."""
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            return torch.flip(x, dims=[-3])  # Depth flip on the third to last dimension (depth)
        return x

import kornia.geometry.transform as K

class RandomElasticDeformation3D(nn.Module):
    def __init__(self, alpha=10, sigma=3, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            orig_shape = x.shape
        
        if x.ndim == 4:
            x = x.unsqueeze(0)
        
            try:
                B, C, D, H, W = x.shape
                device = x.device

                # Create fixed size base grid
                base_grid = F.affine_grid(
                    torch.eye(3, 4, device=device).unsqueeze(0),
                    size=x.shape,
                    align_corners=True
                )

                # Create displacement field
                displacement = torch.randn(
                    (B, D, H, W, 3),
                    device=device
                ) * self.alpha / min(D, H, W)

                # Smooth displacement
                displacement = self._smooth_displacement(displacement)

                # Add to base grid and clamp
                grid = torch.clamp(base_grid + displacement, -1, 1)

                # Apply deformation
                x_deformed = F.grid_sample(
                    x,
                    grid,
                    mode='bilinear',
                    padding_mode='reflection',
                    align_corners=True
                )

                # Ensure output matches input size
                if x_deformed.shape != x.shape:
                    x_deformed = F.interpolate(
                        x_deformed,
                        size=(D, H, W),
                        mode='trilinear',
                        align_corners=True
                    )

                # Restore original shape
                if len(orig_shape) == 4:
                    x_deformed = x_deformed.squeeze(0)

                return x_deformed

            except Exception as e:
                logging.error(f"Elastic deformation error: {str(e)}")
                return x.squeeze(0) if x.ndim == 5 else x

        return x

    def _smooth_displacement(self, displacement):
        """Smooth displacement field with Gaussian filtering."""
        # Apply smoothing separately for each dimension
        for dim in range(3):
            displacement[..., dim] = self._gaussian_filter(
                displacement[..., dim].unsqueeze(1)
            ).squeeze(1)
        return displacement

    def _gaussian_filter(self, tensor):
        """Apply 3D Gaussian smoothing."""
        kernel_size = int(2 * self.sigma) * 2 + 1
        kernel = self._create_gaussian_kernel(kernel_size).to(tensor.device)
        
        padding = kernel_size // 2
        tensor = F.pad(tensor, (padding,)*6, mode='reflect')
        
        # Apply convolution
        tensor = F.conv3d(
            tensor,
            kernel.unsqueeze(0).unsqueeze(0),
            padding=0
        )
        
        return tensor

    def _create_gaussian_kernel(self, size):
        """Create 3D Gaussian kernel."""
        sigma = self.sigma
        coords = torch.arange(size).float() - (size - 1) / 2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g = g / g.sum()
        kernel = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        return kernel

class RandomRotation3D(torch.nn.Module):
    def __init__(self, degrees, p=0.5):
        super().__init__()
        self.degrees = degrees
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            orig_ndim = x.ndim
            if x.ndim == 4:
                x = x.unsqueeze(0)
            
            try:
                # Generate random angles
                yaw = self.rng.uniform(-self.degrees, self.degrees)
                pitch = self.rng.uniform(-self.degrees, self.degrees)
                roll = self.rng.uniform(-self.degrees, self.degrees)

                # Apply rotation 
                x_rotated = K.rotate3d(
                    x,
                    yaw=torch.tensor(yaw, device=x.device),
                    pitch=torch.tensor(pitch, device=x.device),
                    roll=torch.tensor(roll, device=x.device),
                    center=None,
                    mode='bilinear',
                    padding_mode='zeros'
                )

                return x_rotated.squeeze(0) if orig_ndim == 4 else x_rotated

            except Exception as e:
                logging.warning(f"Rotation failed: {str(e)}")
                return x.squeeze(0) if orig_ndim == 4 else x

        return x

class RandomAffine3D(nn.Module):
    def __init__(self, degrees, translate, scale, p=0.5):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            # Ensure 5D input [B,C,D,H,W]
            orig_ndim = x.ndim
            if x.ndim == 4:
                x = x.unsqueeze(0)
            
            B, C, D, H, W = x.shape
            device = x.device

            try:
                #transformation matrix
                yaw = self.rng.uniform(-self.degrees, self.degrees)
                pitch = self.rng.uniform(-self.degrees, self.degrees)
                roll = self.rng.uniform(-self.degrees, self.degrees)

                #Scale matrix
                sx = sy = sz = self.rng.uniform(self.scale[0], self.scale[1])
                
                # Translation
                tx = self.rng.uniform(-self.translate[0], self.translate[0])
                ty = self.rng.uniform(-self.translate[1], self.translate[1])
                tz = self.rng.uniform(-self.translate[2], self.translate[2])

                # Convert to radians
                yaw = np.deg2rad(yaw)
                pitch = np.deg2rad(pitch)
                roll = np.deg2rad(roll)

                #rotation matrices
                Rx = torch.tensor([
                    [1, 0, 0, 0],
                    [0, np.cos(roll), -np.sin(roll), 0],
                    [0, np.sin(roll), np.cos(roll), 0],
                    [0, 0, 0, 1]
                ], device=device, dtype=torch.float32)

                Ry = torch.tensor([
                    [np.cos(pitch), 0, np.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [-np.sin(pitch), 0, np.cos(pitch), 0],
                    [0, 0, 0, 1]
                ], device=device, dtype=torch.float32)

                Rz = torch.tensor([
                    [np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], device=device, dtype=torch.float32)

                # Combine rotations
                R = torch.mm(torch.mm(Rz, Ry), Rx)

                # Add scale and translation
                transform_matrix = torch.eye(4, device=device)
                transform_matrix[:3, :3] = R[:3, :3] * torch.tensor([sx, sy, sz], device=device).view(3, 1)
                transform_matrix[:3, 3] = torch.tensor([tx, ty, tz], device=device)

                # Apply transformation using warp_affine3d
                x_transformed = K.warp_affine3d(
                    x,
                    transform_matrix[:3].unsqueeze(0),  #first 3 rows and add batch dim
                    dsize=(D, H, W),
                    padding_mode='zeros'
                )

                return x_transformed.squeeze(0) if orig_ndim == 4 else x_transformed

            except Exception as e:
                logging.warning(f"Affine transformation failed: {str(e)}")
                return x.squeeze(0) if orig_ndim == 4 else x

        return x


class BayesianLayer3D(nn.Module):
    """Bayesian Layer for uncertainty estimation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.dropout = nn.Dropout3d(Config.DROPOUT_RATE)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_rho, -3)
        nn.init.constant_(self.b_mu, 0)
        nn.init.constant_(self.b_rho, -3)

    def forward(self, x, sample=False):
        if self.training or sample:
            w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(x.device)
            b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(x.device)
            w = self.w_mu + torch.log1p(torch.exp(self.w_rho)) * w_epsilon
            b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * b_epsilon
        else:
          w = self.w_mu
          b = self.b_mu
        
        return self.dropout(torch.matmul(x, w.t()) + b)
    
class SelfAttention3D(nn.Module):
    """Self-Attention Layer."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv3d(in_channels, in_channels // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv3d(in_channels, in_channels // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv3d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        q = self.query(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, depth * height * width)
        v = self.value(x).view(batch_size, -1, depth * height * width)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        attn_output = torch.bmm(v, attn.permute(0, 2, 1)).view(batch_size, channels, depth, height, width)
        
        return x + self.gamma * attn_output
    
class VisionTransformerBlock3D(nn.Module):
    """Vision Transformer Block."""
    def __init__(self, num_channels, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
            nn.Dropout(Config.DROPOUT_RATE)
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, D*H*W, C)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x = x + attn_output.permute(0, 2, 1).view(batch_size, channels, depth, height, width)
        x = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        x = x + self.mlp(x.permute(0, 2, 3, 4, 1).view(batch_size, -1, channels)).view(batch_size, depth, height, width, channels).permute(0, 4, 1, 2, 3)
        x = self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

class UNetWithAttention3D(nn.Module):
    """U-Net with Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv3d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )
        self.attention = SelfAttention3D(num_channels * 2)
        self.up1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(num_channels, in_channels, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x2 = self.attention(x2)
        x = self.up1(x2)
        x = self.up2(x)
        return x

class FeatureExtractor3D(nn.Module):
    """Extracts intermediate features from the ResNet18 model."""
    def __init__(self):
        super().__init__()
        self.model = resnet.resnet18(pretrained=True)
        self.features = nn.ModuleList(list(self.model.children())[:-2]) # Extract features at each stage.
        self.eval()

    def forward(self, x):
        with torch.no_grad():
           x = x.repeat(1, 3, 1, 1, 1) # repeat the grayscale image into 3 channels
           x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]) # flatten depth and width to a 2D tensor
           output_features = []
           for layer in self.features:
                 x = layer(x)
                 output_features.append(x)
           return output_features

class Generator3D(nn.Module):
    """Generator with Vision Transformers and U-Net Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, num_channels, kernel_size=9, padding=4)),
            nn.PReLU()
        )
        
        self.transformer_blocks = nn.Sequential(*[
            VisionTransformerBlock3D(num_channels, num_heads=Config.NUM_HEADS) for _ in range(Config.NUM_RESIDUAL_BLOCKS)
        ])
        
        self.conv_block = nn.Sequential(
             spectral_norm(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1)),
             nn.BatchNorm3d(num_channels),
             spectral_norm(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1)),
             nn.BatchNorm3d(num_channels)
        )
        
        self.upsampling = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(num_channels, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.PReLU(),
            spectral_norm(nn.ConvTranspose3d(num_channels, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.PReLU()
        )
        
        self.final = nn.Sequential(
            spectral_norm(nn.Conv3d(num_channels, in_channels, kernel_size=9, padding=4)),
            nn.Tanh()
        )
    
    def forward(self, x, sample = False, layer_activations = False):
        initial = self.initial(x)
        if layer_activations:
            layer_outputs = [initial]
        x = initial
        for i, layer in enumerate(self.transformer_blocks):
             x = layer(x)
             if layer_activations and (i+1) in Config.ACTIVATION_LAYERS:
                  layer_outputs.append(x)
        x = self.conv_block(x) + initial
        x = self.upsampling(x)
        x = self.final(x)
        if layer_activations:
              return x, layer_outputs
        return x

class Discriminator3D(nn.Module):
    """Discriminator with Spectral Normalization and Self-Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [spectral_norm(nn.Conv3d(in_feat, out_feat, kernel_size=4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm3d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, num_channels, normalize=False),  # 64x64x64 -> 32x32x32
            *discriminator_block(num_channels, num_channels * 2),           # 32x32x32 -> 16x16x16
            *discriminator_block(num_channels * 2, num_channels * 4),        # 16x16x16 -> 8x8x8
            *discriminator_block(num_channels * 4, num_channels * 8),        # 8x8x8 -> 4x4x4
            SelfAttention3D(num_channels * 8), # no shape change
            spectral_norm(nn.Conv3d(num_channels * 8, 1, kernel_size = 4, stride=1, padding=0))#reduce to 1x1x1
        )
       

    def forward(self, x):
       #Add shape check for debugging
       """print("Input shape to discriminator:", x.shape)
       #for layer in self.model:
         x = layer(x)
         print("Shape after layer:", x.shape)"""
       return self.model(x)

class WarmUpScheduler(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warm up scheduler """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 1
    
class Trainer:
    def __init__(self, use_segmentation=False, progressive_growing=False):
        self.setup_directories()
        setup_logging()
        setup_distributed()
        
        self.device = Config.DEVICE
        if Config.DISTRIBUTED:
            self.device = torch.device(f"cuda:{Config.LOCAL_RANK}")
        
        self.scaler = GradScaler(enabled=Config.MIXED_PRECISION)
        self.writer = SummaryWriter(os.path.join(Config.LOG_DIR, 'tensorboard')) if is_main_process() else None
        
        self.generator = Generator3D().to(self.device)
        self.discriminator = Discriminator3D().to(self.device)
        
        if Config.DISTRIBUTED:
            self.generator = DistributedDataParallel(self.generator, device_ids=[Config.LOCAL_RANK], output_device=Config.LOCAL_RANK)
            self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[Config.LOCAL_RANK], output_device=Config.LOCAL_RANK)
        
        self.g_optimizer = RAdam(
            self.generator.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        self.d_optimizer = RAdam(
            self.discriminator.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.g_scheduler = CosineAnnealingWarmRestarts(self.g_optimizer, T_0=10, T_mult=2)
        self.d_scheduler = CosineAnnealingWarmRestarts(self.d_optimizer, T_0=10, T_mult=2)

        self.g_scheduler_plateau = ReduceLROnPlateau(self.g_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.d_scheduler_plateau = ReduceLROnPlateau(self.d_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.g_scheduler_cyclic = CyclicLR(self.g_optimizer, base_lr = Config.CYCLE_LR_BASE_LR, max_lr = Config.CYCLE_LR_MAX_LR, step_size_up=Config.CYCLE_LR_STEP_SIZE_UP, mode = Config.CYCLE_LR_MODE)
        self.d_scheduler_cyclic = CyclicLR(self.d_optimizer, base_lr = Config.CYCLE_LR_BASE_LR, max_lr = Config.CYCLE_LR_MAX_LR, step_size_up=Config.CYCLE_LR_STEP_SIZE_UP, mode = Config.CYCLE_LR_MODE)

        self.g_warmup_scheduler = WarmUpScheduler(self.g_optimizer, Config.WARMUP_STEPS)
        self.d_warmup_scheduler = WarmUpScheduler(self.d_optimizer, Config.WARMUP_STEPS)

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_content = nn.L1Loss()
        self.lpips = LPIPS(net='alex', verbose=False).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        
        self.feature_extractor = FeatureExtractor3D().to(self.device)
        self.feature_extractor.eval()
        
        self.best_val_metric = float('inf')
        self.epochs_without_improvement = 0
        self.dice = Dice().to(self.device)
        self.jaccard = JaccardIndex(num_classes=1, task="binary").to(self.device)

        self.use_segmentation = use_segmentation
        self.progressive_growing = progressive_growing

        self.integrated_gradients = IntegratedGradients(self.generator)
        self.current_image_size = Config.LOW_RES_SIZE * 4
    
    def setup_directories(self):
        if is_main_process():
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(Config.GENERATED_DIR, exist_ok=True)
            os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    def train_step(self, lr_images, hr_images, seg_images = None, total_steps = None):
        if lr_images.ndim == 4:
            lr_images = lr_images.unsqueeze(1)
        batch_size = lr_images.size(0)
        real_label = torch.ones(batch_size, 1, 1, 1, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 1, 1, 1).to(self.device)
        
        # Train Discriminator
        self.discriminator.zero_grad()
        with autocast(device_type=self.device.type, enabled=Config.MIXED_PRECISION):
            fake_images = self.generator(lr_images)
            d_real = self.discriminator(hr_images)
            d_fake = self.discriminator(fake_images.detach())
            
            d_real_loss = self.criterion_gan(d_real, real_label)
            d_fake_loss = self.criterion_gan(d_fake, fake_label)
            d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        
        # Train Generator
        self.generator.zero_grad()
        with autocast(device_type=self.device.type, enabled=Config.MIXED_PRECISION):
            g_fake = self.discriminator(fake_images)
            g_gan_loss = self.criterion_gan(g_fake, real_label)
            g_content_loss = self.criterion_content(fake_images, hr_images)
            g_perceptual_loss = self.calculate_perceptual_loss(fake_images, hr_images)
            g_ssim_loss = 1 - self.ssim(fake_images, hr_images)
            
            hr_features = self.feature_extractor(hr_images)
            sr_features = self.feature_extractor(fake_images)
            g_feature_loss = self.criterion_content(sr_features[0], hr_features[0]) # only using the first layer

            if self.use_segmentation and seg_images is not None:
                seg_loss = self.criterion_content(fake_images, seg_images)
                g_loss = (
                    Config.LAMBDA_CONTENT * g_content_loss +
                    Config.LAMBDA_ADV * g_gan_loss +
                    g_perceptual_loss +
                    Config.LAMBDA_SSIM * g_ssim_loss +
                    Config.LAMBDA_FEATURE * g_feature_loss +
                    Config.SEG_WEIGHT * seg_loss
                )
            else:
                 g_loss = (
                    Config.LAMBDA_CONTENT * g_content_loss +
                    Config.LAMBDA_ADV * g_gan_loss +
                    g_perceptual_loss +
                    Config.LAMBDA_SSIM * g_ssim_loss +
                    Config.LAMBDA_FEATURE * g_feature_loss
                )
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        torch.cuda.empty_cache()
        
        metrics = {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_content_loss': g_content_loss.item(),
            'g_perceptual_loss': g_perceptual_loss.item(),
            'g_ssim_loss': g_ssim_loss.item(),
            'g_feature_loss' : g_feature_loss.item(),
            'g_psnr': self.psnr(fake_images, hr_images).item(),
            'g_dice': self.dice(fake_images > 0.5, hr_images > 0.5).item(),
            'g_jaccard': self.jaccard(fake_images > 0.5, hr_images > 0.5).item()
        }

        if self.use_segmentation and seg_images is not None:
            metrics['seg_loss'] = seg_loss.item()
        
        if is_main_process():
          for key, value in metrics.items():
              self.writer.add_scalar(key, value, total_steps)

        return metrics
    
    def calculate_perceptual_loss(self, fake_images, hr_images):
        perceptual_loss = 0
        hr_features = self.feature_extractor(hr_images)
        sr_features = self.feature_extractor(fake_images)
        for i, (hr_feat, sr_feat, weight) in enumerate(zip(hr_features, sr_features, Config.PERC_LAYER_WEIGHTS)):
               sr_feat = sr_feat.view(sr_feat.shape[0], sr_feat.shape[1], -1)
               hr_feat = hr_feat.view(hr_feat.shape[0], hr_feat.shape[1], -1)
               perceptual_loss += weight * self.criterion_content(sr_feat, hr_feat)
        return Config.LAMBDA_PERC * perceptual_loss

    def save_checkpoint(self, epoch, losses, is_final=False):
        """Save checkpoint with proper state dict handling."""
        if is_main_process():
            try:
                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': self.generator.module.state_dict() if Config.DISTRIBUTED else self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.module.state_dict() if Config.DISTRIBUTED else self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'g_scheduler_state_dict': self.g_scheduler.state_dict(),
                    'd_scheduler_state_dict': self.d_scheduler.state_dict(),
                    'losses': losses
                }
                
                path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth') if not is_final else os.path.join(Config.CHECKPOINT_DIR, 'final_checkpoint.pth')
                torch.save(checkpoint, path)
                logging.info(f"Saved checkpoint: {path}")
                
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {str(e)}")
    
    def save_sample_images(self, epoch, lr_images, hr_images, sr_images, is_final=False):
        """Save medical image visualizations with proper anatomical views."""
        if is_main_process():
            try:
                # Move to CPU and detach
                lr_images = lr_images.cpu().detach()
                hr_images = hr_images.cpu().detach()
                sr_images = sr_images.cpu().detach()

                # Get central slices for each view
                def get_orthogonal_slices(volume):
                    D, H, W = volume.shape[-3:]
                    axial = volume[..., D//2, :, :]      # Top view
                    sagittal = volume[..., :, H//2, :]   # Side view
                    coronal = volume[..., :, :, W//2]    # Front view
                    return axial, sagittal, coronal

                # Create figure with subplots
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                
                # Plot each view for each volume
                for idx, (title, volume) in enumerate([
                    ('LR', lr_images[0,0]),
                    ('SR', sr_images[0,0]),
                    ('HR', hr_images[0,0])
                ]):
                    axial, sagittal, coronal = get_orthogonal_slices(volume)
                    
                    # Plot axial view
                    axes[0,idx].imshow(axial, cmap='gray')
                    axes[0,idx].set_title(f'{title} Axial')
                    axes[0,idx].axis('off')
                    
                    # Plot sagittal view
                    axes[1,idx].imshow(sagittal, cmap='gray')
                    axes[1,idx].set_title(f'{title} Sagittal')
                    axes[1,idx].axis('off')
                    
                    # Plot coronal view
                    axes[2,idx].imshow(coronal, cmap='gray')
                    axes[2,idx].set_title(f'{title} Coronal')
                    axes[2,idx].axis('off')

                plt.tight_layout()
                
                # Save path
                save_path = os.path.join(
                    Config.GENERATED_DIR,
                    f'epoch_{epoch}_views.png' if not is_final else 'final_generated_views.png'
                )

                # Save figure
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

                logging.info(f"Saved orthogonal views to {save_path}")

            except Exception as e:
                logging.error(f"Error saving sample images: {str(e)}")

    def train(self, train_loader, num_epochs, val_loader = None):
            try:
                total_steps = 0
                
                for epoch in range(1, num_epochs + 1):
                    self.generator.train()
                    self.discriminator.train()
                    
                    epoch_losses = []
                    if Config.DISTRIBUTED:
                       train_loader.sampler.set_epoch(epoch)
                    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', disable=not is_main_process())
                    
                    vis_batch = next(iter(train_loader))
                    if self.use_segmentation:
                        vis_lr, vis_hr, _ = vis_batch
                    else:
                        vis_lr, vis_hr = vis_batch
                        
                    vis_lr = vis_lr.to(self.device)
                    vis_hr = vis_hr.to(self.device)

                    for batch_idx, batch in enumerate(progress_bar):
                        if self.use_segmentation:
                           lr_images, hr_images, seg_images = batch
                           seg_images = seg_images.to(self.device)
                        else:
                           lr_images, hr_images = batch
                           seg_images = None

                        lr_images = lr_images.to(self.device)
                        hr_images = hr_images.to(self.device)
                        
                        losses = self.train_step(lr_images, hr_images, seg_images, total_steps)
                        epoch_losses.append(losses)
                        
                        progress_bar.set_postfix(
                            d_loss=f"{losses['d_loss']:.4f}",
                            g_loss=f"{losses['g_loss']:.4f}"
                        )
                        
                        total_steps += 1

                        self.g_scheduler_cyclic.step()
                        self.d_scheduler_cyclic.step()

                        # Learning rate warmup
                        self.g_warmup_scheduler.step()
                        self.d_warmup_scheduler.step()

                        if batch_idx % 100 == 0:  # Adjust frequency as needed
                            self.generator.eval()
                            with torch.no_grad():
                                # Generate SR image
                                fake_images = self.generator(vis_lr)
                                
                                # Save images
                                if is_main_process():
                                    self.save_sample_images(
                                        epoch,
                                        vis_lr,
                                        vis_hr, 
                                        fake_images,
                                        is_final=(epoch == num_epochs and batch_idx == len(train_loader)-1)
                                    )
                            self.generator.train()
                    
                    avg_losses = {
                        key: sum(loss[key] for loss in epoch_losses) / len(epoch_losses)
                        for key in epoch_losses[0].keys()
                    }
                    
                    if is_main_process():
                        logging.info(f"Epoch {epoch} Average Losses:")
                        for key, value in avg_losses.items():
                            logging.info(f"{key}: {value:.4f}")

                    if val_loader is not None:
                        val_metrics = self.validate(val_loader, epoch)
                        if is_main_process():
                            logging.info(f"Epoch {epoch} Validation Metrics:")
                            for key, value in val_metrics.items():
                                logging.info(f"{key}: {value:.4f}")

                            current_val_metric = val_metrics['val_ssim'] # Using ssim as a metric for early stopping
                            if current_val_metric > self.best_val_metric + Config.EARLY_STOPPING_MIN_DELTA:
                                    self.best_val_metric = current_val_metric
                                    self.epochs_without_improvement = 0
                            else:
                                    self.epochs_without_improvement += 1
                            
                            if self.epochs_without_improvement > Config.EARLY_STOPPING_PATIENCE:
                                logging.info(f"Early stopping triggered at epoch {epoch}")
                                break

                    
                    if epoch % 5 == 0 or epoch == num_epochs: # save checkpoint at the end of training
                        self.save_checkpoint(epoch, avg_losses)

                    #generating sample images
                    if is_main_process():
                        self.generator.eval()
                        with torch.no_grad():
                            fake_images = self.generator(vis_lr)
                            if is_main_process():
                                self.save_sample_images(
                                    epoch,
                                    vis_lr,
                                    vis_hr,
                                    fake_images,
                                    is_final=(epoch == num_epochs)
                                )
                        self.generator.train()

                    # Reduce Learning Rate on plateau
                    if val_loader is not None:
                      self.g_scheduler_plateau.step(avg_losses['g_loss'])
                      self.d_scheduler_plateau.step(avg_losses['d_loss'])

                    self.g_scheduler.step()
                    self.d_scheduler.step()
                
                # Save final model and visualizations
                if is_main_process():
                      self.save_checkpoint(epoch, avg_losses, is_final=True)
                      self.generator.eval()
                      with torch.no_grad():
                          if self.use_segmentation:
                              sample_lr, sample_hr, _ = next(iter(train_loader))
                          else:
                              sample_lr, sample_hr = next(iter(train_loader))
                          sample_lr = sample_lr[:1].to(self.device)
                          sample_hr = sample_hr[:1].to(self.device)
                          sample_sr = self.generator(sample_lr)
                          self.save_sample_images(epoch, sample_lr, sample_hr, sample_sr, is_final = True)
                      self.generator.train()


            except Exception as e:
                logging.error(f"Training error: {str(e)}")
                raise
            finally:
                if is_main_process() and self.writer:
                    self.writer.close()
    def validate(self, val_loader, epoch):
        self.generator.eval()
        all_metrics = []
        with torch.no_grad():
           for batch in tqdm(val_loader, desc = f"Validating Epoch {epoch}", disable = not is_main_process()):
                 if self.use_segmentation:
                     lr_images, hr_images, seg_images = batch
                     seg_images = seg_images.to(self.device)
                 else:
                     lr_images, hr_images = batch
                     seg_images = None
                 
                 lr_images = lr_images.to(self.device)
                 hr_images = hr_images.to(self.device)
                 
                 metrics = self.calculate_metrics(lr_images, hr_images, seg_images)
                 all_metrics.append(metrics)
        avg_metrics = {
            key: sum(metrics[key] for metrics in all_metrics) / len(all_metrics) for key in all_metrics[0].keys()
        }
        self.generator.train()
        return avg_metrics
    
    def calculate_metrics(self, lr_images, hr_images, seg_images = None):
         with torch.no_grad():
            sr_images = self.generator(lr_images)
            psnr = self.psnr(sr_images, hr_images)
            ssim = self.ssim(sr_images, hr_images)
            dice = self.dice(sr_images > 0.5, hr_images > 0.5)
            jaccard = self.jaccard(sr_images > 0.5, hr_images > 0.5)
            content_loss = self.criterion_content(sr_images, hr_images)

            metrics = {
                'val_psnr': psnr.item(),
                'val_ssim': ssim.item(),
                'val_dice': dice.item(),
                'val_jaccard': jaccard.item(),
                'val_content_loss' : content_loss.item()
            }

            if self.use_segmentation and seg_images is not None:
               seg_loss = self.criterion_content(sr_images, seg_images)
               metrics['val_seg_loss'] = seg_loss.item()

            return metrics

    def predict(self, lr_image, num_samples=10, post_process = True):
        self.generator.eval()
        with torch.no_grad():
            if lr_image.ndim == 4: # If its a single image (c,d,h,w)
              lr_image = lr_image.unsqueeze(0)
            
            lr_image = lr_image.to(self.device)
            
            if num_samples > 1: # Enable monte carlo dropout
              sr_images = torch.stack([self.generator(lr_image, sample = True) for _ in range(num_samples)])
              sr_image = torch.mean(sr_images, dim = 0)
              uncertainty = torch.var(sr_images, dim = 0) # Variance as uncertainty
              if post_process:
                sr_image = self.post_process(sr_image)

              return sr_image, uncertainty
            else:
              sr_image = self.generator(lr_image)
              if post_process:
                  sr_image = self.post_process(sr_image)
              return sr_image, None # No uncertainty returned

    def post_process(self, image, patch_size = 7, patch_distance = 5, fast_mode = True, h = 0.02):
        """
        Applies non-local means filtering to the given image.
        """
        if isinstance(image, torch.Tensor):
             image = image.squeeze(0).cpu().numpy() # remove batch dim and move to cpu
        
        image = img_as_float(image)
        filtered_image = denoise_nl_means(image, patch_size = patch_size, patch_distance = patch_distance, fast_mode = fast_mode, h=h, channel_axis=0)

        return torch.tensor(filtered_image, dtype = torch.float32).unsqueeze(0).to(self.device)
    
    def calculate_activations(self, lr_images):
          self.generator.eval()
          with torch.no_grad():
             _, layer_outputs = self.generator(lr_images, layer_activations = True)
             return layer_outputs
    
    def visualize_activations(self, lr_images, layer_outputs):
         if is_main_process():
           num_layers = len(layer_outputs)
           num_images = lr_images.shape[0]
           fig, axes = plt.subplots(num_layers, num_images, figsize=(15, 4 * num_layers))

           if num_layers == 1:
              axes = [axes]

           for i, activations in enumerate(layer_outputs):
             for j in range(num_images):
                 ax = axes[i][j] if num_layers > 1 else axes[j]
                 activation_map = activations[j].mean(dim=0).cpu().numpy()
                 ax.imshow(activation_map, cmap='viridis')
                 ax.axis('off')
                 ax.set_title(f"Image {j+1} Layer {i+1}")

           plt.tight_layout()
           plt.savefig(os.path.join(Config.GENERATED_DIR, 'activation_maps.png'))

    def calculate_integrated_gradients(self, lr_images, hr_images):
       self.generator.eval()
       lr_images = lr_images.to(self.device)
       hr_images = hr_images.to(self.device)

       integrated_grads = self.integrated_gradients.attribute(lr_images, target = hr_images)

       return integrated_grads
    
    def visualize_integrated_gradients(self, lr_images, integrated_grads):
       if is_main_process():
         num_images = lr_images.shape[0]
         fig, axes = plt.subplots(1, num_images, figsize=(15, 4))

         if num_images == 1:
             axes = [axes]
         
         for j in range(num_images):
             ax = axes[j] if num_images > 1 else axes
             grads = integrated_grads[j].mean(dim=0).cpu().numpy()
             ax.imshow(grads, cmap = "viridis")
             ax.axis('off')
             ax.set_title(f"Image {j+1} Integrated Gradients")

         plt.tight_layout()
         plt.savefig(os.path.join(Config.GENERATED_DIR, 'integrated_gradients.png'))

def create_data_loaders(dataset, fold_idx, num_folds):
    # Calculate indices for validation set
    fold_size = len(dataset) // num_folds
    start_idx = fold_idx * fold_size
    end_idx = (fold_idx + 1) * fold_size
    val_indices = list(range(start_idx, end_idx))

    # Split the remaining dataset into training set indices
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]

    # Create samplers 
    train_sampler = RandomSampler(train_indices) if not Config.DISTRIBUTED else dist.DistributedSampler(train_indices, shuffle=True)
    val_sampler = SequentialSampler(val_indices) if not Config.DISTRIBUTED else dist.DistributedSampler(val_indices, shuffle=False)
    
    # Create dataloaders 
    train_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
    
    val_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

    return train_loader, val_loader

def main():
    try:
        setup_logging()
        logging.info("Starting SRGAN training pipeline")
        
        dataset_path = download_brats_dataset()
        if is_main_process():
             Config.set_dataset_folder(dataset_path)
             logging.info(f"Dataset path set to: {dataset_path}")

        use_segmentation = True 
        progressive_growing = True 
        
        if not os.path.exists(Config.TRAIN_HR_FOLDER) or not os.path.exists(Config.TRAIN_LR_FOLDER) or not os.path.exists(Config.VAL_HR_FOLDER) or not os.path.exists(Config.VAL_LR_FOLDER) :
            if is_main_process():
                 logging.info("Dataset not preprocessed. Starting preprocessing...")
                 num_images = preprocess_dataset(use_segmentation = use_segmentation, validation = False, subset = Config.SUBSET_SIZE)
                 num_images_val = preprocess_dataset(use_segmentation = use_segmentation, validation = True, validation_subset = Config.VALIDATION_SUBSET_SIZE)
                 logging.info(f"Preprocessing complete. Created {num_images} training image patches, {num_images_val} validation patches")
            else:
                  logging.info("Waiting for main process to process the dataset...")
                  torch.distributed.barrier()
        else:
            if is_main_process():
                logging.info("Using previously preprocessed dataset, deleting for re-processing...")
                for file in os.listdir(Config.TRAIN_HR_FOLDER):
                   if file.endswith(".pt"):
                       os.remove(os.path.join(Config.TRAIN_HR_FOLDER, file))
                for file in os.listdir(Config.TRAIN_LR_FOLDER):
                    if file.endswith(".pt"):
                       os.remove(os.path.join(Config.TRAIN_LR_FOLDER, file))
                for file in os.listdir(Config.VAL_HR_FOLDER):
                     if file.endswith(".pt"):
                          os.remove(os.path.join(Config.VAL_HR_FOLDER, file))
                for file in os.listdir(Config.VAL_LR_FOLDER):
                     if file.endswith(".pt"):
                          os.remove(os.path.join(Config.VAL_LR_FOLDER, file))
                
                logging.info("Starting re-preprocessing...")
                num_images = preprocess_dataset(use_segmentation = use_segmentation, validation = False, subset = Config.SUBSET_SIZE)
                num_images_val = preprocess_dataset(use_segmentation = use_segmentation, validation = True, validation_subset = Config.VALIDATION_SUBSET_SIZE)
                logging.info(f"Re-preprocessing complete. Created {num_images} training image patches, {num_images_val} validation patches")
            else:
                logging.info("Waiting for main process to re-process the dataset")
                torch.distributed.barrier()

        # Cross Validation Training
        for fold_idx in range(Config.NUM_FOLDS):
            if is_main_process():
                logging.info(f"Starting training for Fold {fold_idx + 1}/{Config.NUM_FOLDS}")

            trainer = Trainer(use_segmentation = use_segmentation, progressive_growing=progressive_growing)
            
            dataset = BraTSDataset(
                    Config.TRAIN_HR_FOLDER,
                    Config.TRAIN_LR_FOLDER,
                    transform=True,
                    use_segmentation = use_segmentation,
                    norm = True,
                    subset = 1 # use all data in the dataloader, subset was already done at preprocessing.
                )
            
            val_dataset = BraTSDataset(
                    Config.VAL_HR_FOLDER,
                    Config.VAL_LR_FOLDER,
                    transform=False, # No need for transformations during validation.
                    use_segmentation = use_segmentation,
                    norm = True,
                    subset = 1 # use all validation data in the dataloader, subset was done at preprocessing.
                )

            train_loader, val_loader = create_data_loaders(dataset, fold_idx, Config.NUM_FOLDS)
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY
                ) # create the dataloader for the validation data.

            trainer.train(train_loader, Config.EPOCHS, val_loader = val_loader)
            

        if is_main_process():
            logging.info("Training completed successfully")
        
        # Predict on a sample image using Monte Carlo Dropout
        if is_main_process():
            sample_dataset = BraTSDataset(
            Config.TRAIN_HR_FOLDER,
            Config.TRAIN_LR_FOLDER,
            transform = False,
            use_segmentation = use_segmentation,
            norm = False,
            subset = 0.05
        )
            
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle = True)
            sample_batch = next(iter(sample_loader))

            if use_segmentation:
                sample_lr, sample_hr, *_ = sample_batch
            else:
                sample_lr, sample_hr = sample_batch

            sr_image, uncertainty = trainer.predict(sample_lr, num_samples = 10) # Get the SR image and uncertainty with mc dropout
            save_image(sr_image.cpu(), os.path.join(Config.GENERATED_DIR, 'sample_sr_image.png'), normalize=True)

            if uncertainty is not None:
                save_image(uncertainty.cpu(), os.path.join(Config.GENERATED_DIR, 'sample_uncertainty.png'), normalize=True)
            logging.info(f"Generated Sample SR Image and saved to {Config.GENERATED_DIR}")
            
            layer_outputs = trainer.calculate_activations(sample_lr.to(trainer.device))
            trainer.visualize_activations(sample_lr, layer_outputs)
            logging.info(f"Generated Sample activation maps and saved to {Config.GENERATED_DIR}")

            integrated_grads = trainer.calculate_integrated_gradients(sample_lr, sample_hr.to(trainer.device))
            trainer.visualize_integrated_gradients(sample_lr, integrated_grads)
            logging.info(f"Generated  ample Integrated Gradients and saved to {Config.GENERATED_DIR}")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        raise
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()