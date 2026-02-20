import numpy as np

def read_avi(video_path, max_frames=None, to_rgb=True):
	"""Read an .avi video into a numpy array.

	Args:
		video_path: Path to the .avi file.
		max_frames: Optional cap on frames to read.
		to_rgb: Convert frames from BGR to RGB if True.

	Returns:
		frames: Numpy array of shape (T, H, W, C) with uint8 dtype.
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Failed to open video: {video_path}")

	frames = []
	count = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if to_rgb:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame)
		count += 1
		if max_frames is not None and count >= max_frames:
			break

	cap.release()
	return np.array(frames)  # (T, H, W, C)

data = np.load()


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import glob
import os

class SapiensDataset(Dataset):
    def __init__(self, img_dir, input_size=(1024, 768)):
        """
        Args:
            img_dir (str): Path to the folder containing images.
            input_size (tuple): Model input size (H, W). 
                                Sapiens usually expects 1024x768 or 1024x1024.
        """
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                         glob.glob(os.path.join(img_dir, "*.png"))
        
        # Standard ImageNet normalization (Required for Sapiens)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(), # Converts [0,255] -> [0.0,1.0]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # 1. LOAD IMAGE (OpenCV loads as BGR by default)
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            # Handle broken images or return a placeholder
            return torch.zeros((3, 1024, 768))

        # 2. CONVERT TO RGB (Critical for Sapiens)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. APPLY TRANSFORMS (Resize, Normalize, ToTensor)
        img_tensor = self.transform(img_rgb)
        
        return img_tensor, img_path

# --- Usage Example ---

# 1. Initialize Dataset
# Replace 'path/to/images' with your actual directory
dataset = SapiensDataset(img_dir='./data/test_images', input_size=(1024, 768))

# 2. Initialize DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

# 3. Iterate
print(f"Total images: {len(dataset)}")

for batch_imgs, batch_paths in dataloader:
    # batch_imgs shape: [Batch_Size, 3, 1024, 768]
    # This tensor is now ready to be fed into the model
    print(f"Batch shape: {batch_imgs.shape}")
    
    # Example Inference:
    # output = sapiens_model(batch_imgs.cuda())
    
    break # Stop after one batch for demonstration