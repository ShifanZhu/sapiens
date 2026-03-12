import numpy as np
import torch
import smplx
import cv2
import os
import pandas as pd


def bedlam_to_visualization_coords(joints):
	"""Convert BEDLAM/Unreal world coordinates to play_joints visualization coordinates.
	
	BEDLAM (Unreal Engine) uses:
		- X: forward
		- Y: right  
		- Z: up
		- Units: cm
	
	play_joints visualization expects:
		- X: forward
		- Y: right (despite misleading comment saying "left")
		- Z: up
	
	Actually, since the SMPL-X model outputs joints in its own coordinate system:
		SMPL-X canonical:
		- X: right
		- Y: up
		- Z: backward
	
	After applying pose_world rotations and trans_world translation from BEDLAM,
	the joints should already be in BEDLAM world coordinates, BUT we need to verify
	if BEDLAM stores pose_world with coordinate transformations baked in.
	
	Based on df_full_body_smplx_bedlam2.py line 272, BEDLAM applies:
		transform_coordinate = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
	This flips Y and Z when converting between coordinate systems.
	
	For now, we'll assume the joints from SMPL-X with pose_world/trans_world are in:
		- X: right (SMPL-X canonical X)
		- Y: up (SMPL-X canonical Y)
		- Z: backward (SMPL-X canonical Z)
	
	And need conversion to visualization (Z-up):
		- X_new = Z (backward → forward via sign flip in visualization)
		- Y_new = X (right)
		- Z_new = Y (up)
	"""
	vis_joints = joints.copy()
	vis_joints[..., 0] = joints[..., 2]  # X_vis = -Z_smplx (forward)
	vis_joints[..., 1] = joints[..., 0]   # Y_vis = X_smplx (right)
	vis_joints[..., 2] = -joints[..., 1]   # Z_vis = Y_smplx (up)
	return vis_joints


def get_smplx_joints_from_npz(npz_path, model_path, use_world=True):
	data = np.load(npz_path)  # Pass None for csv_path since we're not using it in this function
	poses = data["pose_world" if use_world else "pose_cam"][:89]
	trans = data["trans_world" if use_world else "trans_cam"][:89]
	betas = data["shape"][:89]
	gender = data["gender"][:89]
	
	if isinstance(gender, np.ndarray):
		gender = gender.flat[0]
	gender = str(gender).strip().lower()
	if gender not in {"male", "female", "neutral"}:
		gender = "neutral"

	num_frames = poses.shape[0]

	global_orient = torch.tensor(poses[:, :3], dtype=torch.float32)
	body_pose = torch.tensor(poses[:, 3:66], dtype=torch.float32)
	jaw_pose = torch.tensor(poses[:, 66:69], dtype=torch.float32)
	leye_pose = torch.tensor(poses[:, 69:72], dtype=torch.float32)
	reye_pose = torch.tensor(poses[:, 72:75], dtype=torch.float32)
	left_hand_pose = torch.tensor(poses[:, 75:120], dtype=torch.float32)
	right_hand_pose = torch.tensor(poses[:, 120:165], dtype=torch.float32)
	transl = torch.tensor(trans, dtype=torch.float32)  # (num_frames, 1, 3)



	if betas.ndim == 1:
		betas = np.repeat(betas[None, :], num_frames, axis=0)
	betas_t = torch.tensor(betas, dtype=torch.float32)  # (1, num_frames, num_betas)
	print(global_orient.shape, body_pose.shape, jaw_pose.shape, leye_pose.shape, reye_pose.shape, left_hand_pose.shape, right_hand_pose.shape, transl.shape, betas_t.shape)
	breakpoint()
	model = smplx.create(
		model_path=model_path,
		gender=gender,
		model_type='smplx',
		num_betas=16,
		use_pca=False,
		batch_size=num_frames,
		flat_hand_mean=True
	)

	output = model(
		betas=betas_t,
		global_orient=global_orient,
		body_pose=body_pose,
		jaw_pose=jaw_pose,
		leye_pose=leye_pose,
		reye_pose=reye_pose,
		left_hand_pose=left_hand_pose,
		right_hand_pose=right_hand_pose,
		transl=transl,
	)


	return output.joints.detach().cpu().numpy()


def load_video(video_path, max_frames=None):
	"""Load video file and return frames as numpy array.
	
	Args:
		video_path: Path to MP4 or other video file
		max_frames: Maximum number of frames to load (None for all)
	
	Returns:
		frames: NumPy array of shape (nframe, H, W, C) in BGR format
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Cannot open video file: {video_path}")
	
	frames = []
	frame_count = 0
	
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(frame)
		frame_count += 1
		if max_frames is not None and frame_count >= max_frames:
			break
	
	cap.release()
	
	if not frames:
		raise ValueError(f"No frames loaded from video: {video_path}")
	
	return np.array(frames)  # Shape: (nframe, H, W, 3) in BGR


def get_corresponding_video_path(npz_path):
	"""Convert NPZ path to corresponding MP4 video path.
	
	Assumes the video is in the same directory with .mp4 extension.
	Tries multiple naming patterns to find the video.
	"""
	base_path = os.path.splitext(npz_path)[0]
	video_path = base_path + '.mp4'
	
	if os.path.exists(video_path):
		return video_path
	
	# Try alternative naming: replace 'labels' with 'videos' in path
	video_path = npz_path.replace('_labels_processed', '_videos_processed')
	video_path = os.path.splitext(video_path)[0] + '.mp4'
	
	if os.path.exists(video_path):
		return video_path
	
	return None


def load_csv_coordinates(csv_path, max_frames=None):
	"""Load x, y, z coordinates from CSV file.
	
	Args:
		csv_path: Path to CSV file with columns: name, x, y, z, ...
		max_frames: Maximum number of frames to load (None for all)
	
	Returns:
		coords: NumPy array of shape (nframe, 3) containing [x, y, z] coordinates
	"""
	df = pd.read_csv(csv_path)
	
	# Extract x, y, z columns
	coords = df[['x', 'y', 'z']].values.astype(np.float32)
	
	if max_frames is not None:
		coords = coords[:max_frames*5]
		coords = coords[::5] * 0.01  # Sample every 5th frame to match joints
	
	return coords


def combine_joints_with_csv(joints, csv_coords):
	"""Combine SMPL-X joints with CSV coordinates.
	
	Args:
		joints: NumPy array of shape (nframe, njoint, 3) - SMPL-X joints
		csv_coords: NumPy array of shape (nframe, 3) - CSV coordinates
	
	Returns:
		combined: NumPy array of shape (nframe, njoint + 1, 3)
	"""
	# Reshape csv_coords to (nframe, 1, 3) to match joints format
	csv_coords_reshaped = csv_coords[:, np.newaxis, :]
	
	# Concatenate along joint dimension
	combined = np.concatenate([joints, csv_coords_reshaped], axis=1)
	
	return combined


if __name__ == "__main__":
	pth = "/media/hang/8tb-data/datasets/bedlam2_labels_processed/20240416_1_171_yogastudio_orbit_timeofday.npz"
	model_path = "/home/hang/repos_local/MMC/sapiens/smplx/smplx_lockedhead_20230207/models_lockedhead"
	csv_path = '/media/hang/8tb-data/datasets/gt/20240416_1_171_yogastudio_orbit_timeofday_gt_centersubframe_exr_meta_csv/20240416_1_171_yogastudio_orbit_timeofday/ground_truth/meta_exr_csv/seq_000000_camera.csv'
	
	words_joints = get_smplx_joints_from_npz(pth, model_path, use_world=True)
	cam_joints = get_smplx_joints_from_npz(pth, model_path, use_world=False)
	print(f"Joints shape: {words_joints.shape}")
	print(f"Joints range: X [{words_joints[..., 0].min():.2f}, {words_joints[..., 0].max():.2f}], "
	      f"Y [{words_joints[..., 1].min():.2f}, {words_joints[..., 1].max():.2f}], "
	      f"Z [{words_joints[..., 2].min():.2f}, {words_joints[..., 2].max():.2f}]")
	
	# Convert to visualization coordinates
	joints_vis = bedlam_to_visualization_coords(cam_joints)
	
	# Load CSV coordinates and combine with joints
	print(f"\nLoading CSV coordinates from: {csv_path}")
	try:
		csv_coords = load_csv_coordinates(csv_path, max_frames=89)
		print(f"CSV coords shape: {csv_coords.shape}")
		print(f"CSV coords range: X [{csv_coords[:, 0].min():.2f}, {csv_coords[:, 0].max():.2f}], "
		      f"Y [{csv_coords[:, 1].min():.2f}, {csv_coords[:, 1].max():.2f}], "
		      f"Z [{csv_coords[:, 2].min():.2f}, {csv_coords[:, 2].max():.2f}]")
		
		# Combine SMPL-X joints with CSV coordinates
		combined_joints = combine_joints_with_csv(joints_vis[:89], csv_coords)
		print(f"Combined joints shape: {combined_joints.shape}")
		
	except Exception as e:
		print(f"Error loading CSV: {e}")
		combined_joints = joints_vis[:89]
		print("Using SMPL-X joints only")
	
	from play_joints import play_joints
	
	# Load video if available
	video_path = '/media/hang/8tb-data/datasets/bedlam2_download/20240416_1_171_yogastudio_orbit_timeofday_mp4/20240416_1_171_yogastudio_orbit_timeofday/mp4/seq_000000.mp4'
	if os.path.exists(video_path):
		print(f"\nLoading video from: {video_path}")
		try:
			video_frames = load_video(video_path, max_frames=89*5)
			video_frames = video_frames[::5]  # Sample every 5th frame to match joints
			print(f"Video frames shape: {video_frames.shape}")
			# Display joints with video side-by-side
			play_joints(combined_joints, frame_interval=1/6, video_frames=video_frames)
		except Exception as e:
			print(f"Error loading video: {e}")
			print("Falling back to joints-only visualization")
			play_joints(combined_joints, frame_interval=1/6)
	else:
		print("No video found, showing joints only")
		play_joints(combined_joints, frame_interval=1/6)
	