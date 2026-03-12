import numpy as np
from utils import visualize_joints_with_video, load_video, project_joints_on_video
from getlable import bones

label = np.load('/home/hang/repos_local/MMC/sapiens/label/20240425_1_171_citysample_dolly/seq_000001.npz', allow_pickle=True)


joints_world = label['joints_world'][0]  # (n_body, n_frames, n_joints, 3)
joints_cam = label['joints_cam'][0]      # (n_body, n_frames,
joints_2d = label['joints_2d'][0]      # (n_body, n_frames, n_joints, 2)
joints_2d_mask = label['joints_2d_mask'][0]  # (n_body, n_frames, n_joints)
intrinsic_matrix = label['intrinsic_matrix']  # (3, 3)
camera_position = label['camera_position']  # (n_frames, 3)
camera_rotation = label['camera_rotation']  # (n_frames, 3) 
seq_name = label['seq_name']
folder_name = label['folder_name']
scene_name = label['scene_name']
n_frames = label['n_frames']
n_joints = label['n_joints']
rotate_flag = label['rotate_flag']

mp4_parent_folder = '/media/hang/8tb-data/datasets/bedlam2_download'
mp4_folder = f'{mp4_parent_folder}/{folder_name}_mp4/{folder_name}/mp4'
mp4_path = f'{mp4_folder}/{seq_name}.mp4'
#video_frames = load_video(mp4_path, max_frames=n_frames, rotate_flag=rotate_flag)
#video_frames_with_joints = project_joints_on_video(video_frames, joints_cam, intrinsic_matrix, fps=6.0, bones=bones, joint_radius=3, bone_thickness=1)
visualize_joints_with_video(
    joints_world,
    joints_cam,
    video_frames=np.zeros((n_frames, 720, 1280, 3), dtype=np.uint8),  # Placeholder black video
    fps=6.0,
    view_elev=0.0,
    view_azim=180.0,
    bones=bones,
)

print(joints_world.shape)