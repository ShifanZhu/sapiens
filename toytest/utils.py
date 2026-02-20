import os

import numpy as np
import cv2
from play_joints import play_joints
from checkdata import read_avi

def generate_data_path(npy_data_folder_path) -> str:

    parent_folder = os.path.dirname(npy_data_folder_path.rstrip('/'))
    npy_files = os.listdir(npy_data_folder_path)
    npy_files = sorted(npy_files, key=lambda x: int(x.split('C')[0][1:]))  # Sort by index in filename
    with open('toytest/data_paths.txt', 'w') as f:
        for npy_file in npy_files:
            idx = npy_file.split('.')[0]
            npy_file_path = os.path.join(npy_data_folder_path, npy_file)
            sid = int(idx[1:4])
            depth_path = f'{parent_folder}/nturgbd_depth_masked_s{sid:03d}/nturgb+d_depth_masked_s{sid:03d}/nturgb+d_depth_masked/{idx}'
            rgb_path = f'{parent_folder}/nturgbd_rgb_s{sid:03d}/nturgb+d_rgb_s{sid:03d}/nturgb+d_rgb/{idx}_rgb.avi'
            f.write(f'{rgb_path} {depth_path} {npy_file_path}\n')

#generate_data_path('/home/hang/data/NTU-RGBD/npy_skeletons_s018_to_s032')

def read_data_paths(file_path: str):
    data_paths = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            rgb_path, depth_path, npy_path = line.split(' ')
            data_paths.append((rgb_path, depth_path, npy_path))
    return data_paths

def read_skeleton(npy_path):
    return np.load(npy_path, allow_pickle=True).item()


def read_exr_depth_frame(fpath):
    depth = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    depth = depth[:,:,:3]
    cv2.imshow("Depth Frame", depth)
    cv2.waitKey(0)  # Display the frame for a brief moment
    if depth is not None:
        return depth



def read_exr_depth_sequence(depth_dir, max_frames=None):
    exr_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(".exr")]
    exr_files.sort()
    if max_frames is not None:
        exr_files = exr_files[:max_frames]

    frames = []
    for fname in exr_files:
        fpath = os.path.join(depth_dir, fname)
        depth = read_exr_depth_frame(fpath)
        frames.append(depth)

    if not frames:
        raise ValueError(f"No EXR files found in: {depth_dir}")

    return np.stack(frames, axis=0)


def play_exr_depth_video(depth_dir, fps=30, max_frames=None, window_name="Depth"):
    depth_frames = read_exr_depth_sequence(depth_dir, max_frames=max_frames)
    delay_ms = max(1, int(1000 / fps))

    for depth in depth_frames:
        if depth.ndim == 3:
            depth = depth[..., 0]

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        d_min = float(np.min(depth))
        d_max = float(np.max(depth))
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.float32)

        depth_u8 = (depth_norm * 255.0).astype(np.uint8)
        if depth_u8.ndim == 3 and depth_u8.shape[2] == 1:
            depth_u8 = depth_u8[:, :, 0]
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

        cv2.imshow(window_name, depth_color)
        key = cv2.waitKey(delay_ms)
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(window_name)

import OpenEXR
import Imath
import numpy as np

def read_ue5_depth_exr(file_path):
    # 1. Open the EXR file
    if not OpenEXR.isOpenExrFile(file_path):
        raise ValueError(f"File {file_path} is not a valid EXR file.")
        
    exr_file = OpenEXR.InputFile(file_path)
    
    # Optional: Print all available channels in the EXR to debug
    # print("Available channels:", exr_file.header()['channels'].keys())
    
    # 2. Get the resolution dynamically from the EXR header (should be 1280x720)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 3. Define the exact channel name and the 16-bit float type (HALF)
    channel_name = "FinalImageMovieRenderQueue_WorldDepth.R"
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)
    
    try:
        # 4. Extract the raw byte data from that specific layer
        raw_bytes = exr_file.channel(channel_name, pixel_type)
    except TypeError:
        # Fallback: If MRQ didn't use the long name, it might just be 'R'
        print(f"Warning: '{channel_name}' not found. Falling back to standard 'R' channel.")
        raw_bytes = exr_file.channel('R', pixel_type)
        
    # 5. Convert the byte buffer into a numpy array (float16)
    depth_img = np.frombuffer(raw_bytes, dtype=np.float16)
    
    # 6. Reshape to (H, W) -> (720, 1280)
    depth_img = depth_img.reshape((height, width))
    
    # 7. Convert to float32 for safer math/processing downstream
    return depth_img.astype(np.float32)

# --- Usage in your utils.py ---
# fpath = '/media/hang/8tb-data/datasets/depth1/.../seq_000000/0000.exr'
# depth_frame = read_ue5_depth_exr(fpath)
# print(f"Depth shape: {depth_frame.shape}, Max depth: {np.max(depth_frame)}")

if __name__ == '__main__':
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img = read_ue5_depth_exr('/media/hang/8tb-data/datasets/depth1/20240806_1_250_ai1101_vcam_exr_depth.0/20240806_1_250_ai1101_vcam/exr_depth/seq_000000/seq_000000_0000.exr')
    img = img / 500
    cv2.imshow("UE5 Depth", img)
    cv2.waitKey(0)
    # play_exr_depth_video('/media/hang/8tb-data/datasets/depth1/20240806_1_250_ai1101_vcam_exr_depth.0/20240806_1_250_ai1101_vcam/exr_depth/seq_000000/', fps=6, max_frames=300)
    # path = read_data_paths('toytest/data_paths.txt')
    # for i in range(50):
        
    #     rgb_path, depth_path, npy_path = path[i]
        
    #     data = read_skeleton(npy_path)
    #     ndodys = data['nbodys'][0]
    #     skel_body = data['skel_body0']
    #     if ndodys > 0:
    #         for j in range(1, ndodys):
    #             skel_bodyx = data[f'skel_body{j}']
    #             skel_body = np.concatenate((skel_body, skel_bodyx), axis=1)  # (T, 25, 3)

        
    #     print(skel_body.shape)  # (T, 25, 3)
        
    #     # Load video frames
    #     video = read_avi(rgb_path, to_rgb=True)
    #     print(f"Video shape: {video.shape}")  # (T, H, W, C)
        
    #     # Play skeleton with video side-by-side
    #     play_joints(skel_body, video_frames=video)

    

    # import numpy as np
    # import torch
    # import smplx

    # # 1) Load AMASS-style NPZ
    # pth = '/media/hang/8tb-data/datasets/b2_motions_npz_training/motions_npz_training'
    # files = os.listdir(pth)

    # pth = '/media/hang/8tb-data/datasets/bedlam2_labels_processed/20250219_3-4_250_middleeast_vcam_approach.npz'
    # motion_pth = '/media/hang/8tb-data/datasets/b2_motions_npz_training/motions_npz_training/us_1726_3XL_2400.npz'
    # gt_data = np.load(pth, allow_pickle=True)

    # data = np.load('/media/hang/8tb-data/datasets/b2_motions_npz_training/motions_npz_training/us_1726_3XL_2400.npz', allow_pickle=True)    

    # poses = data["poses"]        # (N, 165)
    # trans = data["trans"]         # (N, 3)
    # betas = data["betas"]        # (num_betas,)
    # num_betas = betas.shape[0]  # e.g., 10
    # gender = str(data["gender"])  # 'male', 'female', or 'neutral'

    # num_frames = poses.shape[0]

    # # 2) Split the 165-dim poses into SMPL-X components
    # #    global_orient:    3
    # #    body_pose:       63  (21 joints × 3)
    # #    jaw_pose:         3
    # #    leye_pose:        3
    # #    reye_pose:        3
    # #    left_hand_pose:  45  (15 joints × 3)
    # #    right_hand_pose: 45  (15 joints × 3)
    # global_orient   = torch.tensor(poses[:, :3],      dtype=torch.float32)
    # body_pose       = torch.tensor(poses[:, 3:66],     dtype=torch.float32)
    # jaw_pose        = torch.tensor(poses[:, 66:69],    dtype=torch.float32)
    # leye_pose       = torch.tensor(poses[:, 69:72],    dtype=torch.float32)
    # reye_pose       = torch.tensor(poses[:, 72:75],    dtype=torch.float32)
    # left_hand_pose  = torch.tensor(poses[:, 75:120],   dtype=torch.float32)
    # right_hand_pose = torch.tensor(poses[:, 120:165],  dtype=torch.float32)
    # betas_t         = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(num_frames, -1)
    # transl          = torch.tensor(trans, dtype=torch.float32)

    # # 3) Create SMPL-X model
    # # model_path should be the PARENT folder containing a "smplx/" subfolder
    # model = smplx.create(
    #     model_path="/home/hang/repos_local/MMC/sapiens/smplx/smplx_lockedhead_20230207/models_lockedhead",
    #     model_type="smplx",
    #     gender=gender,
    #     num_betas=num_betas,
    #     use_pca=False,
    #     batch_size=num_frames
    # )

    # # 4) Forward pass
    # output = model(
    #     betas=betas_t,
    #     global_orient=global_orient,
    #     body_pose=body_pose,
    #     jaw_pose=jaw_pose,
    #     leye_pose=leye_pose,
    #     reye_pose=reye_pose,
    #     left_hand_pose=left_hand_pose,
    #     right_hand_pose=right_hand_pose,
    #     transl=transl
    # )

    # joints = output.joints.detach().numpy()  # (N, 127, 3) — all joints
    # vertices = output.vertices.detach().numpy()  # (N, 10475, 3) — mesh vertices

    # print(f"Joints shape: {joints.shape}")
    # print(f"Frame 0, joint 0 (pelvis): {joints[0, 0]}")

    # breakpoint()
        
        # SMPL-X/AMASS coordinate system: Y-up (Y is vertical), X is right, Z is backward
        # Convert to visualization coordinate system: Z-up, X is forward, Y is left
        # Transformation: new_X = old_Z, new_Y = old_X, new_Z = old_Y
        # joints_vis = joints.copy()
        # joints_vis[..., 0] = joints[..., 2]  # X_new = Z_old (forward/backward)
        # joints_vis[..., 1] = joints[..., 0]  # Y_new = X_old (left/right)
        # joints_vis[..., 2] = joints[..., 1]  # Z_new = Y_old (up/down)
        
        # play_joints(joints_vis)