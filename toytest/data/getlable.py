import pandas as pd
import os
import numpy as np
import cv2
from utils import smplx_forward, world_coords_to_camera, human_coords_to_world, visualize_joints_with_video, load_video, project_joints_on_video, compute_intrinsic_matrix
from utils import get_smplx_skeleton, project_joints_to_2d, cam_to_right_handed, get_smplx_model, project_mesh_on_video
from matplotlib import pyplot as plt
scene_names_csv = 'toytest/data/bedlam2_scene_names.csv'
scene_names_csv = pd.read_csv(scene_names_csv)

folder_name = scene_names_csv['Folder name'].tolist()
scene_names = scene_names_csv['Scene name'].str.strip().tolist()

smplx_model_path = "/home/hang/repos_local/MMC/sapiens/smplx/smplx_lockedhead_20230207/models_lockedhead"

print("Folder names:")
print(folder_name)
print("Scene names:")
print(scene_names)

gt_parent_folder = '/media/hang/8tb-data/datasets/gt'
motion_parent_folder = '/media/hang/8tb-data/datasets/b2_motions_npz_training/motions_npz_training'
mp4_parent_folder = '/media/hang/8tb-data/datasets/bedlam2_download'

bones = get_smplx_skeleton()


def visualize_camera(cam_x, cam_y, cam_z, cam_pitch, cam_roll, cam_yaw, title=None, fps=30.0, animate=True, marker_x=None, marker_y=None):
	"""Visualize camera position and orientation over time using OpenCV.
	
	Args:
		marker_x, marker_y: Optional x, y coordinates of a reference point to display on the XY trajectory (as a green circle).
	"""
	cam_x = np.asarray(cam_x, dtype=np.float32)
	cam_y = np.asarray(cam_y, dtype=np.float32)
	cam_z = np.asarray(cam_z, dtype=np.float32)
	cam_pitch = np.asarray(cam_pitch, dtype=np.float32)
	cam_roll = np.asarray(cam_roll, dtype=np.float32)
	cam_yaw = np.asarray(cam_yaw, dtype=np.float32)

	if len(cam_x) == 0:
		return

	width, height = 1200, 600
	left_w = width // 2
	right_w = width - left_w
	margin = 40

	def scale(v, vmin, vmax, out_min, out_max):
		if vmax - vmin < 1e-6:
			return int((out_min + out_max) * 0.5)
			
		return int(out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin))

	x_min, x_max = cam_x.min(), cam_x.max()
	y_min, y_max = cam_y.min(), cam_y.max()
	ang_min = min(cam_pitch.min(), cam_roll.min(), cam_yaw.min())
	ang_max = max(cam_pitch.max(), cam_roll.max(), cam_yaw.max())

	frames = len(cam_x)
	frame_delay = int(max(1, 1000.0 / fps))

	for i in range(frames):
		canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

		# Left panel: XY trajectory
		pts = []
		for j in range(i + 1):
			px = scale(cam_x[j], x_min, x_max, margin, left_w - margin)
			py = scale(cam_y[j], y_min, y_max, height - margin, margin)
			pts.append([px, py])
		if len(pts) >= 2:
			cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (255, 0, 0), 2)
		origin_x = scale(0.0, x_min, x_max, margin, left_w - margin)
		origin_y = scale(0.0, y_min, y_max, height - margin, margin)
		
		# Draw X and Y axes at origin
		axis_length = 60
		cv2.arrowedLine(canvas, (origin_x, origin_y), (origin_x + axis_length, origin_y), (0, 0, 255), 2, tipLength=0.15)  # X-axis (red)
		cv2.arrowedLine(canvas, (origin_x, origin_y), (origin_x, origin_y - axis_length), (0, 255, 0), 2, tipLength=0.15)  # Y-axis (green)
		cv2.putText(canvas, 'X', (origin_x + axis_length + 5, origin_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(canvas, 'Y', (origin_x + 5, origin_y - axis_length - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		
		cv2.circle(canvas, (origin_x, origin_y), 4, (0, 0, 0), -1)
		cv2.putText(canvas, '0,0', (origin_x + 6, origin_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
		if marker_x is not None and marker_y is not None:
			mark_x = scale(marker_x, x_min, x_max, margin, left_w - margin)
			mark_y = scale(marker_y, y_min, y_max, height - margin, margin)
			cv2.circle(canvas, (mark_x, mark_y), 5, (0, 255, 0), 2)
			cv2.putText(canvas, f'({marker_x:.1f},{marker_y:.1f})', (mark_x + 8, mark_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
		cur_x = scale(cam_x[i], x_min, x_max, margin, left_w - margin)
		cur_y = scale(cam_y[i], y_min, y_max, height - margin, margin)
		cv2.circle(canvas, (cur_x, cur_y), 5, (0, 0, 255), -1)
		cv2.putText(canvas, 'Camera XY', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

		# Right panel: angles over time
		def plot_angle(series, color):
			pts = []
			for j in range(i + 1):
				px = scale(j, 0, frames - 1, left_w + margin, width - margin)
				py = scale(series[j], ang_min, ang_max, height - margin, margin)
				pts.append([px, py])
			if len(pts) >= 2:
				cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

		plot_angle(cam_pitch, (0, 0, 255))
		plot_angle(cam_roll, (0, 128, 0))
		plot_angle(cam_yaw, (255, 0, 0))

		cur_t = scale(i, 0, frames - 1, left_w + margin, width - margin)
		cv2.line(canvas, (cur_t, margin), (cur_t, height - margin), (0, 0, 0), 1)
		cv2.putText(canvas, 'pitch', (left_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.putText(canvas, 'roll', (left_w + 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)
		cv2.putText(canvas, 'yaw', (left_w + 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
		
		# Display current angle values
		cv2.putText(canvas, f'{cam_pitch[i]:.1f}°', (left_w + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(canvas, f'{cam_roll[i]:.1f}°', (left_w + 90, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
		cv2.putText(canvas, f'{cam_yaw[i]:.1f}°', (left_w + 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

		if title:
			cv2.putText(canvas, title, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		cv2.imshow('Camera Visualization', canvas)
		key = cv2.waitKey(frame_delay) & 0xFF
		if key == ord('q'):
			break

	if not animate:
		cv2.waitKey(0)
	cv2.destroyAllWindows()




def get_joint(start_frame, smplx_param_orig, trans_body, rotate_body, cam_x, cam_y, cam_z,
               cam_yaw_=0., cam_pitch_=0., cam_roll_=0., smplx_model_path=None, return_right_handed=True):
      
	# Saving every 5th frame to match 6fps video frames
	total_frames = np.ceil(len(cam_x) / 5)
	total_frames = int(total_frames)
	smplx_ind = [start_frame + i * 5 for i in range(total_frames)]
	cam_ind = [i * 5 for i in range(total_frames)]
	cam_x = np.array(cam_x)[cam_ind]
	cam_y = np.array(cam_y)[cam_ind]
	cam_z = np.array(cam_z)[cam_ind]
	cam_pitch_ = np.array(cam_pitch_)[cam_ind]
	cam_roll_ = np.array(cam_roll_)[cam_ind]
	cam_yaw_ = np.array(cam_yaw_)[cam_ind]
	pose = smplx_param_orig['poses'][smplx_ind]
	transl = smplx_param_orig['trans'][smplx_ind]
	beta = smplx_param_orig['betas'][:16]
	gender = smplx_param_orig['gender']
	
	
 
	vertices_right_handed, joints_human_coords_right_handed = smplx_forward(smplx_model_path, gender, pose, beta, transl)
	joints_human_coords = joints_human_coords_right_handed.copy()
	joints_human_coords[:, :, 1] = joints_human_coords_right_handed[:, :, 2]
	joints_human_coords[:, :, 2] = joints_human_coords_right_handed[:, :, 1]
	vertices = vertices_right_handed.copy()
	vertices[:, :, 1] = vertices_right_handed[:, :, 2]
	vertices[:, :, 2] = vertices_right_handed[:, :, 1]
	joints_world_coords = human_coords_to_world(joints_human_coords, trans_body[0], trans_body[1], trans_body[2], rotate_body[0], rotate_body[1], rotate_body[2])	
	joints_cam_coords = world_coords_to_camera(joints_world_coords, cam_x, cam_y, cam_z, cam_yaw_, cam_pitch_, cam_roll_)
	vertices_world_coords = human_coords_to_world(vertices, trans_body[0], trans_body[1], trans_body[2], rotate_body[0], rotate_body[1], rotate_body[2])
	vertices_cam_coords = world_coords_to_camera(vertices_world_coords, cam_x, cam_y, cam_z, cam_yaw_, cam_pitch_, cam_roll_)
	if return_right_handed:
		joints_cam_coords[:, :, 1] = -joints_cam_coords[:, :, 1]
		joints_world_coords[:, :, 1] = -joints_world_coords[:, :, 1]
		vertices_cam_coords[:, :, 1] = -vertices_cam_coords[:, :, 1]
		vertices_world_coords[:, :, 1] = -vertices_world_coords[:, :, 1]
	return joints_world_coords, joints_cam_coords, vertices_world_coords, vertices_cam_coords


		
if __name__ == '__main__':

	joints_2d_coords_list, joints_2d_mask_list, joints_cam_list, joints_world_list = [], [], [], []

	for i in range(len(folder_name)):
		print(f"Processing folder: {folder_name[i]}, scene: {scene_names[i]}")
		mp4_folder = f'{mp4_parent_folder}/{folder_name[i]}_mp4/{folder_name[i]}/mp4'
		gt_folder = f"{gt_parent_folder}/{folder_name[i]}_gt_centersubframe_exr_meta_csv/{folder_name[i]}"
		be_seq = f'{gt_folder}/be_seq.csv'
		be_seq = pd.read_csv(be_seq).to_dict('list')
		cam_csv_base = os.path.join(gt_folder,'ground_truth/meta_exr_csv')
		
		if 'portrait' in scene_names[i]:
			rotate_flag = True
		else:
			rotate_flag = False
		SENSOR_W = 36
		SENSOR_H = 20.25
		IMG_W = 1280
		IMG_H = 720
		
		for idx, comment in enumerate(be_seq['Comment']):
			if 'sequence_name' in comment:
				#Get sequence name and corresponding camera details
				n_body = 0
				joints_2d_coords_list, joints_2d_mask_list, joints_cam_list, joints_world_list = [], [], [], []
				gender_list, beta_list = [], []
				vertices_world_list, vertices_cam_list = [], []
				seq_name = comment.split(';')[0].split('=')[-1]
				cam_csv_data = pd.read_csv(os.path.join(cam_csv_base, seq_name+'_camera.csv'))
				cam_csv_data = cam_csv_data.to_dict('list')
				cam_x = np.array(cam_csv_data['x']) * 0.01  # Convert from cm to m
				cam_y = np.array(cam_csv_data['y']) * 0.01  # Convert from cm to m
				cam_z = np.array(cam_csv_data['z']) * 0.01  # Convert from cm to m
				cam_yaw_ = cam_csv_data['yaw']
				cam_pitch_ = cam_csv_data['pitch']
				cam_roll_ = cam_csv_data['roll']
				cam_x_r, cam_y_r, cam_z_r, cam_yaw_r, cam_pitch_r, cam_roll_r = cam_to_right_handed(cam_x, cam_y, cam_z, cam_yaw_, cam_pitch_, cam_roll_)
				fl = cam_csv_data['focal_length']
				sw = cam_csv_data['sensor_width']
				sh = cam_csv_data['sensor_height']
				intrinsic_matrix = compute_intrinsic_matrix(focal_length=fl[0], sensor_width=SENSOR_W, sensor_height=SENSOR_H, img_width=IMG_W, img_height=IMG_H)
				continue

			elif 'start_frame' in comment:
				# Get body details
				start_frame = int(comment.split(';')[0].split('=')[-1])
				body = be_seq['Body'][idx]
				n_body += 1
				if 'moyo' in scene_names[i]:
					smplx_param_orig_path = os.path.join(motion_parent_folder, body+'.npz')
				else:
					parts = body.rsplit('_', 1)
					smplx_param_orig_path = os.path.join(motion_parent_folder, parts[0]+'_'+parts[1]+'.npz')
				if not os.path.exists(smplx_param_orig_path):
					print(f'{body} is a test subject. Skipping')
					continue
				smplx_param_orig = np.load(smplx_param_orig_path)
				gender = smplx_param_orig['gender'].item()
				gender_list.append(gender)
				X = be_seq['X'][idx] * 0.01  # Convert from cm to m
				Y = be_seq['Y'][idx] * 0.01  # Convert from cm to m
				Z = be_seq['Z'][idx] * 0.01  # Convert from cm to m
				trans_body = np.array([X, Y, Z])
				yaw_body_ = be_seq['Yaw'][idx]
				pitch_body_ = be_seq['Pitch'][idx]
				roll_body_ = be_seq['Roll'][idx]
				rotate_body = np.array([yaw_body_, pitch_body_, roll_body_])
				#print(cam_z)
		

				joints_world, joints_cam, vertices_world, vertices_cam = get_joint(start_frame, smplx_param_orig, trans_body, rotate_body, cam_x, cam_y, cam_z,
				cam_pitch_=cam_pitch_, cam_roll_=cam_roll_, cam_yaw_=cam_yaw_,
				smplx_model_path=smplx_model_path, return_right_handed=True)
			
				# while True:
				# 	visualize_camera(
				# 	cam_x,
				# 	np.array(cam_y),
				# 	cam_z,
				# 	cam_pitch_,
				# 	cam_roll_,
				# 	cam_yaw_,
				# 	title=seq_name,	
				# 	marker_x=X,
				# 	marker_y=Y
				# )
				# mp4_path = f'{mp4_folder}/{seq_name}.mp4'
				# video_frames = load_video(mp4_path, max_frames=len(cam_x)//5, rotate_flag=rotate_flag)
				# faces = get_smplx_model(smplx_model_path, gender=gender).faces
				# video_frames_with_joints = project_joints_on_video(video_frames, joints_cam, intrinsic_matrix, fps=6.0, bones=bones, joint_radius=3, bone_thickness=1)
				# video_frames_with_vertices = project_mesh_on_video(video_frames, vertices_cam, faces, intrinsic_matrix, fps=6.0)
				joints_2d_coords, joints_2d_mask = project_joints_to_2d(joints_cam, intrinsic_matrix) # [n_frames, n_joints, 2], [n_frames, n_joints], x is right, y is down

				joints_world_list.append(joints_world)
				joints_cam_list.append(joints_cam)
				joints_2d_coords_list.append(joints_2d_coords)	
				joints_2d_mask_list.append(joints_2d_mask)
				vertices_world_list.append(vertices_world)
				vertices_cam_list.append(vertices_cam)
				beta_list.append(smplx_param_orig['betas'][:16])
				

    
				# visualize_joints_with_video(
				# 	world_joint_frame=joints_world,
				# 	cam_joint_frame=joints_cam,
				# 	video_frames_1=video_frames_with_joints,
				# 	video_frames_2=video_frames_with_vertices,   # or None to reuse video1
				# 	bones=bones,
				# 	world_vertices_frame=vertices_world,
				# 	cam_vertices_frame=vertices_cam,
				# 	mesh_faces=faces,     # optional; if None -> vertex scatter
				# 	fps=6,
				# 	save_path=f'demo2.mp4',
				# )
				
    
			if (idx == len(be_seq['Comment']) - 1 or 'sequence_name' in be_seq['Comment'][idx+1]) and len(joints_world_list) > 0:
				label ={
					'folder_name': folder_name[i],
					'scene_name': scene_names[i],
					'seq_name': seq_name,
					'joints_world': np.stack(joints_world_list, axis=0),  # [n_body, n_frames, n_joints, 3]
					'joints_cam': np.stack(joints_cam_list, axis=0),      # [n_body, n_frames, n_joints, 3]
					'joints_2d': np.stack(joints_2d_coords_list, axis=0), # [n_body, n_frames, n_joints, 2]
					'joints_2d_mask': np.stack(joints_2d_mask_list, axis=0), # [n_body, n_frames, n_joints]
					'intrinsic_matrix': intrinsic_matrix, # [3, 3]
					'camera_position': np.stack([cam_x, cam_y, cam_z], axis=1), # [n_frames, 3]
					'camera_rotation': np.stack([cam_pitch_, cam_roll_, cam_yaw_], axis=1), # [n_frames, 3]
					'rotate_flag': rotate_flag,
					'n_body': n_body,
					'n_frames': joints_cam.shape[0],
					'n_joints': joints_cam.shape[1],
					'betas': np.stack(beta_list, axis=0), # [n_body, 16]
					'gender': np.stack(gender_list, axis=0), # [n_body]
	#				'vertices_world': np.stack(vertices_world_list, axis=0), # [n_body, n_frames, n_vertices, 3]
	#				'vertices_cam': np.stack(vertices_cam_list, axis=0), # [n_body, n_frames, n_vertices, 3]
				}
				os.makedirs(f'label/{folder_name[i]}', exist_ok=True)
				np.savez(f'label/{folder_name[i]}/{seq_name}.npz', **label)
				print(f"Finished processing {folder_name[i]}/{seq_name} with body {body}. joints_cam.shape[0]: {joints_cam.shape[0]}")

		