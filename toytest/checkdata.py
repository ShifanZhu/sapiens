path = '/home/hang/data/NTU-RGBD/nturgbd_depth_masked_s003/nturgb+d_depth_masked/S003C001P001R001A024/MDepth-00000050.png'

import cv2
import matplotlib.pyplot as plt
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





if __name__ == '__main__':
# viedo = read_avi('/home/hang/data/NTU-RGBD/nturgbd_rgb_s001/nturgb+d_rgb/S001C001P001R001A001_rgb.avi', to_rgb=False

	pth = '/media/hang/8tb-data/datasets/b2_motions_npz_training/motions_npz_training/it_4001_XL_2400.npz'

	data = np.load(pth, allow_pickle=True)

	pose_world = data['poses'].reshape(-1, 55, 3)  # (T, J, 3)
	pose_cam = data['pose_cam']  # (T, J, 3)
	shape = data['shape']  # (T, 10)