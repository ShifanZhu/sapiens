from pyexpat import model

import cv2
import numpy as np
import torch

def load_video(video_path, max_frames=None, to_rgb=True, fps=6, rotate_flag=False):
    """Load video file and return frames as numpy array.
    
    Args:
        video_path: Path to MP4 or other video file
        max_frames: Maximum number of frames to load (None for all)
        to_rgb: Whether to convert BGR frames to RGB (default True)
        fps: Target FPS for video (default 6)
    
    Returns:
        frames: NumPy array of shape (nframe, H, W, C) in BGR or RGB format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_count = 0
    max_frames = max_frames * (30 // fps) if max_frames is not None else None  # Adjust max_frames for target FPS
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_flag:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(frame)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    frames = np.array(frames[::30//fps])  # Shape: (nframe, H, W, 3) in BGR
    
    if to_rgb:
        frames = frames[..., ::-1]  # Convert BGR to RGB
    
    return frames

def get_smplx_model(model_path, batch_size=1, gender='neutral'):
    import smplx
    gender = str(gender).strip().lower()
    if gender not in {"male", "female", "neutral"}:
        gender = "neutral"
    smplx_model = smplx.create(
        model_path=model_path,
        gender=gender,
        model_type='smplx',
        num_betas=16,
        use_pca=False,
        batch_size=batch_size,
        flat_hand_mean=True
    )
    return smplx_model

def smplx_forward(models_path, gender, pose, beta, trans):
    import torch
    
    model = get_smplx_model(models_path, batch_size=pose.shape[0], gender=gender)
    betas = torch.tensor(beta).float()
    global_orient = torch.tensor(pose[:, :3]).float()
    body_pose = torch.tensor(pose[:, 3:66]).float()
    left_hand_pose = torch.tensor(pose[:, 75:120]).float()
    right_hand_pose = torch.tensor(pose[:, 120:165]).float()
    jaw_pose = torch.tensor(pose[:, 66:69]).float()
    leye_pose = torch.tensor(pose[:, 69:72]).float()
    reye_pose = torch.tensor(pose[:, 72:75]).float()
    transl = torch.tensor(trans).float()
    
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], transl.shape[0], axis=0)
    
    output = model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                transl=transl,openpose_smplx=False,
                )
    return output['vertices'].detach().cpu().numpy(), output['joints'].detach().cpu().numpy()
def world_coords_to_camera(coords, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, cam_roll):
    # Create rotation matrix from camera angles
    R = rotate_matrix(np.radians(cam_yaw), np.radians(cam_pitch), np.radians(cam_roll))
    
    # Translate world coordinates to camera-centered coordinates
    camera_coords = np.array([cam_x, cam_y, cam_z]).T #(nframe, 3)
    camera_coords = np.expand_dims(camera_coords, axis=1)  # Shape (nframe, 1, 3)
    translated = coords - camera_coords
    
    # Rotate to align with camera orientation
    camera_coords = translated @ np.transpose(R, axes=(0, 2, 1))  # Rotate the translated coordinates by the inverse of R (R.T)
    return camera_coords

def human_coords_to_world(coords, human_x, human_y, human_z, human_yaw, human_pitch, human_roll):
    # Create rotation matrix from camera angles
    R = rotate_matrix(np.radians(human_yaw), np.radians(human_pitch), np.radians(human_roll))
    
    # Rotate human coordinates to align with world orientation
    rotated = coords @ R  # Rotate the human coordinates by R
    
    # Translate back to world coordinates
    world_coords = rotated + np.array([human_x, human_y, human_z]).T
    return world_coords
def rotate_matrix(yaw, pitch, roll):
    # unreal engine uses yaw-pitch-roll order, so we apply in reverse order: roll, then pitch, then yaw
    # x-forward, y-right, z-up coordinate system, left-handed
    # 
    cosyaw = np.cos(yaw)
    sinyaw = np.sin(yaw)
    cospitch = np.cos(pitch)
    sinpitch = np.sin(pitch)
    cosroll = np.cos(roll)
    sinroll = np.sin(roll)
    try:
        frames = cosyaw.shape[0]
    except:
        frames = 1
    yaw_rotation = np.zeros((frames, 3, 3))
    yaw_rotation[:, 0, 0] = cosyaw
    yaw_rotation[:, 0, 1] = sinyaw
    yaw_rotation[:, 1, 0] = -sinyaw
    yaw_rotation[:, 1, 1] = cosyaw
    yaw_rotation[:, 2, 2] = 1
    pitch_rotation = np.zeros((frames, 3, 3))
    pitch_rotation[:, 0, 0] = cospitch
    pitch_rotation[:, 0, 2] = sinpitch
    pitch_rotation[:, 1, 1] = 1
    pitch_rotation[:, 2, 0] = -sinpitch
    pitch_rotation[:, 2, 2] = cospitch
    roll_rotation = np.zeros((frames, 3, 3))
    roll_rotation[:, 0, 0] = 1
    roll_rotation[:, 1, 1] = cosroll
    roll_rotation[:, 1, 2] = -sinroll
    roll_rotation[:, 2, 1] = sinroll
    roll_rotation[:, 2, 2] = cosroll
    
    # yaw_rotation = np.array([
    # 	[ np.cos(yaw), np.sin(yaw), 0],
    # 	[-np.sin(yaw), np.cos(yaw), 0],
    # 	[          0,          0,   1]])
    
    # pitch_rotation = np.array([
    # 	[np.cos(pitch), 0, np.sin(pitch)],
    # 	[          0, 1,          0],
    # 	[-np.sin(pitch), 0, np.cos(pitch)]])
    
    # roll_rotation = np.array([
    # 	[1,          0,           0],
    # 	[0, np.cos(roll), -np.sin(roll)],
    # 	[0, np.sin(roll),  np.cos(roll)]])
    
    return roll_rotation @ pitch_rotation @ yaw_rotation





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter



def visualize_joints_with_video(
    world_joint_frame: np.ndarray,
    cam_joint_frame: np.ndarray,
    video_frames: np.ndarray,
    bones: list[tuple[int, int]] | None = None,
    stride: int = 1,
    save_path: str | None = None,
    fps: int = 25,
    view_elev: float = 10.0,
    view_azim: float = -90.0,
    axis_range: float = 2.0,
):
    """
    Visualize 3D joints (world & camera) with RGB video, centered at origin (0, 0, 0).
    Includes a ground plane at z=0.

    Args:
        world_joint_frame: [n_frames, n_joints, 3] - world coordinates
        cam_joint_frame: [n_frames, n_joints, 3] - camera coordinates
        video_frames: [n_frames, H, W, 3] - RGB video frames (uint8)
        bones: list of (i, j) joint pairs to draw skeleton
        stride: show every `stride`-th frame
        save_path: output video path (.mp4 or .gif), None to display
        fps: frames per second
        view_elev: matplotlib 3D view elevation angle
        view_azim: matplotlib 3D view azimuth angle
        axis_range: distance from origin in each direction
    """
    if world_joint_frame.shape != cam_joint_frame.shape:
        raise ValueError("world_joint_frame and cam_joint_frame must have same shape")
    if world_joint_frame.shape[0] != video_frames.shape[0]:
        raise ValueError("Joint frames and video frames must have same count")

    data_world = world_joint_frame[::stride]
    data_cam = cam_joint_frame[::stride]
    video = video_frames[::stride]
    n_frames, n_joints, _ = data_world.shape

    fig = plt.figure(figsize=(16, 5))
    
    # Left: video frame
    ax_img = fig.add_subplot(131)
    ax_img.axis("off")
    im_display = ax_img.imshow(video[0])
    ax_img.set_title("RGB Video")
    
    # Middle: 3D joints in world coords, centered at origin
    ax_world = fig.add_subplot(132, projection="3d")
    ax_world.set_xlabel("X")
    ax_world.set_ylabel("Y")
    ax_world.set_zlabel("Z")
    ax_world.set_xlim(-axis_range, axis_range)
    ax_world.set_ylim(-axis_range, axis_range)
    ax_world.set_zlim(-axis_range, axis_range)
    ax_world.set_box_aspect((1, 1, 1))
    ax_world.view_init(elev=view_elev, azim=view_azim)
    ax_world.set_title("World Joints")

    # Right: 3D joints in camera coords, centered at origin
    ax_cam = fig.add_subplot(133, projection="3d")
    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Y")
    ax_cam.set_zlabel("Z")
    ax_cam.set_xlim(-axis_range, axis_range)
    ax_cam.set_ylim(-axis_range, axis_range)
    ax_cam.set_zlim(-axis_range, axis_range)
    ax_cam.set_box_aspect((1, 1, 1))
    ax_cam.view_init(elev=view_elev, azim=view_azim)
    ax_cam.set_title("Camera Joints")

    # Draw ground plane at z=0
    xx, yy = np.meshgrid(np.linspace(-axis_range, axis_range, 10),
                         np.linspace(-axis_range, axis_range, 10))
    zz = np.zeros_like(xx)
    ax_world.plot_surface(xx, yy, zz, alpha=0.2, color="gray")
    ax_cam.plot_surface(xx, yy, zz, alpha=0.2, color="gray")

    # Draw origin (0, 0, 0) in world coords
    ax_world.scatter([0], [0], [0], s=50, c="red", marker="x", linewidths=3, label="Origin")
    ax_world.legend()

    # Draw origin (0, 0, 0) in camera coords
    ax_cam.scatter([0], [0], [0], s=50, c="red", marker="x", linewidths=3, label="Origin")
    ax_cam.legend()

    # Scatter for world joints
    scat_world = ax_world.scatter(
        data_world[0, :, 0], data_world[0, :, 1], data_world[0, :, 2],
        s=20, c="tab:blue", marker="o"
    )

    # Scatter for camera joints
    scat_cam = ax_cam.scatter(
        data_cam[0, :, 0], data_cam[0, :, 1], data_cam[0, :, 2],
        s=20, c="tab:orange", marker="o"
    )

    # Lines for bones (world)
    bone_lines_world = []
    if bones:
        for i, j in bones:
            line, = ax_world.plot(
                [data_world[0, i, 0], data_world[0, j, 0]],
                [data_world[0, i, 1], data_world[0, j, 1]],
                [data_world[0, i, 2], data_world[0, j, 2]],
                c="k", lw=1.5
            )
            bone_lines_world.append(line)

    # Lines for bones (camera)
    bone_lines_cam = []
    if bones:
        for i, j in bones:
            line, = ax_cam.plot(
                [data_cam[0, i, 0], data_cam[0, j, 0]],
                [data_cam[0, i, 1], data_cam[0, j, 1]],
                [data_cam[0, i, 2], data_cam[0, j, 2]],
                c="k", lw=1.5
            )
            bone_lines_cam.append(line)

    def update(f):
        # Update video frame
        frame_rgb = video[f].astype(np.uint8)
        im_display.set_array(frame_rgb)

        # Update world joints
        pts_world = data_world[f]
        scat_world._offsets3d = (pts_world[:, 0], pts_world[:, 1], pts_world[:, 2])

        if bones:
            for line, (i, j) in zip(bone_lines_world, bones):
                line.set_data(
                    [pts_world[i, 0], pts_world[j, 0]],
                    [pts_world[i, 1], pts_world[j, 1]]
                )
                line.set_3d_properties([pts_world[i, 2], pts_world[j, 2]])

        # Update camera joints
        pts_cam = data_cam[f]
        scat_cam._offsets3d = (pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2])

        if bones:
            for line, (i, j) in zip(bone_lines_cam, bones):
                line.set_data(
                    [pts_cam[i, 0], pts_cam[j, 0]],
                    [pts_cam[i, 1], pts_cam[j, 1]]
                )
                line.set_3d_properties([pts_cam[i, 2], pts_cam[j, 2]])

        ax_img.set_title(f"RGB Video | frame {f+1}/{n_frames}")

        artists = [im_display, scat_world, scat_cam] + bone_lines_world + bone_lines_cam
        return artists

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000/fps, blit=False, repeat=False
    )
    plt.tight_layout()

    if save_path:
        if save_path.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        elif save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError("save_path must end with .mp4 or .gif")
        anim.save(save_path, writer=writer, dpi=100)
        print(f"Saved: {save_path}")
    else:
        plt.show()

def project_joints_to_2d(
    joints_cam: np.ndarray,
    intrinsic_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D camera-space joints to 2D image coordinates.
    
    Args:
        joints_cam: [n_frames, n_joints, 3] or [n_joints, 3] 
                   in (x=forward, y=left, z=up) coordinates
        intrinsic_matrix: [3, 3] camera intrinsic matrix
    
    Returns:
        joints_2d: [n_frames, n_joints, 2] or [n_joints, 2] - 2D pixel coordinates - x is right, y is down
        valid_mask: [n_frames, n_joints] or [n_joints] - True if joint is in front of camera
    """
    single_frame = False
    if joints_cam.ndim == 2:
        joints_cam = joints_cam[np.newaxis, ...]  # [1, n_joints, 3]
        single_frame = True
    
    n_frames, n_joints, _ = joints_cam.shape
    joints_2d_all = np.zeros((n_frames, n_joints, 2), dtype=np.float32)
    valid_mask_all = np.zeros((n_frames, n_joints), dtype=bool)
    
    for f in range(n_frames):
        joints_3d = joints_cam[f]  # [n_joints, 3]
        
        # Convert to OpenCV coordinates: (x=forward, y=left, z=up) -> (x=right, y=down, z=forward)
        joints_cv = np.zeros_like(joints_3d)
        joints_cv[:, 0] = -joints_3d[:, 1]  # x_right = -y_left
        joints_cv[:, 1] = -joints_3d[:, 2]  # y_down = -z_up
        joints_cv[:, 2] = joints_3d[:, 0]   # z_fwd = x_forward
        
        # Check which joints are in front of camera
        valid_mask = joints_cv[:, 2] > 0
        
        # Project 3D to 2D: [u, v, 1]^T = K @ [x, y, z]^T
        joints_homo = joints_cv.T  # [3, n_joints]
        proj_homo = intrinsic_matrix @ joints_homo  # [3, n_joints]
        
        # Normalize by depth
        joints_2d = proj_homo[:2] / (proj_homo[2] + 1e-8)  # [2, n_joints]
        joints_2d = joints_2d.T  # [n_joints, 2]
        
        joints_2d_all[f] = joints_2d
        valid_mask_all[f] = valid_mask
    
    if single_frame:
        return joints_2d_all[0], valid_mask_all[0]
    
    return joints_2d_all, valid_mask_all

def project_joints_on_video(
    video_frames: np.ndarray,
    joints_cam: np.ndarray,
    intrinsic_matrix: np.ndarray,
    bones: list[tuple[int, int]] | None = None,
    joint_color: tuple = (0, 255, 0),
    bone_color: tuple = (255, 0, 0),
    joint_radius: int = 5,
    bone_thickness: int = 2,
    stride: int = 1,
    save_path: str | None = None,
    fps: int = 25,
    return_2d_coords: bool = False,
):
    """
    Project 3D joints onto 2D video frames.
    
    Args:
        video_frames: [n_frames, H, W, 3] - RGB video frames
        joints_cam: [n_frames, n_joints, 3] - 3D joints in camera coords
        intrinsic_matrix: [3, 3] - camera intrinsic matrix
        bones: list of (i, j) joint pairs for skeleton
        joint_color: (B, G, R) color for joints
        bone_color: (B, G, R) color for bones
        joint_radius: radius of joint circles
        bone_thickness: thickness of bone lines
        stride: subsample frames
        save_path: output video path
        fps: frames per second
        return_2d_coords: if True, also return 2D coordinates and valid mask
    
    Returns:
        projected_frames: video with joints drawn
        If return_2d_coords=True: (projected_frames, joints_2d, valid_mask)
    """
    data_video = video_frames[::stride].copy()
    data_joints = joints_cam[::stride]
    n_frames = data_video.shape[0]

    if data_video.shape[0] != data_joints.shape[0]:
        raise ValueError("video_frames and joints_cam must have same number of frames")

    # Use project_joints_to_2d to get all 2D coordinates
    joints_2d_all, valid_mask_all = project_joints_to_2d(data_joints, intrinsic_matrix)

    projected_frames = []

    for f in range(n_frames):
        frame = data_video[f].copy()
        joints_2d = joints_2d_all[f].astype(int)  # [n_joints, 2]
        valid_mask = valid_mask_all[f]  # [n_joints]

        if not valid_mask.any():
            print(f"Warning: Frame {f} - all joints behind camera")
            projected_frames.append(frame)
            continue

        h, w = frame.shape[:2]

        # Draw bones
        if bones:
            for i, j in bones:
                # Only draw if both joints are valid and in frame
                if valid_mask[i] and valid_mask[j]:
                    pt1 = tuple(joints_2d[i])
                    pt2 = tuple(joints_2d[j])
                    
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        cv2.line(frame, pt1, pt2, bone_color, bone_thickness)

        # Draw joints
        for idx, pt in enumerate(joints_2d):
            if valid_mask[idx] and 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, tuple(pt), joint_radius, joint_color, -1)

        if f == 0:
            print(f"Frame 0 - Valid joints: {valid_mask.sum()}/{len(valid_mask)}")
            print(f"2D proj range - x: [{joints_2d[:, 0].min()}, {joints_2d[:, 0].max()}], y: [{joints_2d[:, 1].min()}, {joints_2d[:, 1].max()}]")

        projected_frames.append(frame)

    projected_frames = np.array(projected_frames)

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = projected_frames[0].shape[:2]
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        
        for frame in projected_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Saved: {save_path}")

    if return_2d_coords:
        return projected_frames, joints_2d_all, valid_mask_all
    
    return projected_frames

def compute_intrinsic_matrix(
    focal_length: float,
    sensor_width: float,
    sensor_height: float,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Compute camera intrinsic matrix from focal length and sensor dimensions.

    Args:
        focal_length: focal length in mm
        sensor_width: sensor width in mm
        sensor_height: sensor height in mm
        img_width: image width in pixels (after rotation if applicable)
        img_height: image height in pixels (after rotation if applicable)
        rotate_flag: True if camera was rotated 90° clockwise

    Returns:
        K: [3, 3] intrinsic matrix adjusted for rotation
    """

    # Normal case: no rotation
    pixel_size_x = sensor_width / img_width
    pixel_size_y = sensor_height / img_height
    
    fx = focal_length / pixel_size_x
    fy = focal_length / pixel_size_y
    
    cx = img_width / 2.0
    cy = img_height / 2.0


    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K

def get_smplx_skeleton():
# 1. CORE BODY (Indices 0 - 21)
    body = [
        # Spine and Head
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), 
        
        # Left Leg
        (0, 1), (1, 4), (4, 7), (7, 10), 
        
        # Right Leg
        (0, 2), (2, 5), (5, 8), (8, 11), 
        
        # Left Arm
        (9, 13), (13, 16), (16, 18), (18, 20), 
        
        # Right Arm
        (9, 14), (14, 17), (17, 19), (19, 21)
    ]

    # 2. JAW AND EYES (Indices 15, 22 - 24)
    jaw_eyes = [
        (15, 22), # Head to Jaw
        (15, 23), # Head to Left Eye
        (15, 24)  # Head to Right Eye
    ]

    # 3. LEFT HAND KINEMATICS (Indices 20, 25 - 39)
    left_hand = [
        # Index Finger
        (20, 25), (25, 26), (26, 27),
        # Middle Finger
        (20, 28), (28, 29), (29, 30),
        # Pinky Finger
        (20, 31), (31, 32), (32, 33),
        # Ring Finger
        (20, 34), (34, 35), (35, 36),
        # Thumb
        (20, 37), (37, 38), (38, 39)
    ]

    # 4. RIGHT HAND KINEMATICS (Indices 21, 40 - 54)
    right_hand = [
        # Index Finger
        (21, 40), (40, 41), (41, 42),
        # Middle Finger
        (21, 43), (43, 44), (44, 45),
        # Pinky Finger
        (21, 46), (46, 47), (47, 48),
        # Ring Finger
        (21, 49), (49, 50), (50, 51),
        # Thumb
        (21, 52), (52, 53), (53, 54)
    ]

    # 5. FINGERTIPS (From the Extra Landmarks: Indices 76 - 85 depending on config, 
    # but in standard 127-joint SMPL-X, fingertips map to the last 10 points)
    # Note: If your visualization looks broken here, your specific dataset 
    # might use a different surface-landmark mapping for indices 55-126.
    # Standard SMPL-X appends Left Tips then Right Tips.
    
    # Left fingertips connected to the last kinematic knuckle
    left_tips = [(27, 76), (30, 77), (33, 78), (36, 79), (39, 80)] # (Approximated standard indices)
    
    # Right fingertips connected to the last kinematic knuckle
    right_tips = [(42, 81), (45, 82), (48, 83), (51, 84), (54, 85)] 

    # Combine all rigid kinematic bones
    kinematic_skeleton = body + jaw_eyes + left_hand + right_hand
    
    return kinematic_skeleton

def get_smplx_skeleton_simple():
    """
    Get simplified SMPL-X skeleton (body only, no hands/face).
    """
    bones = [
        # Spine
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        
        # Left leg
        (0, 1), (1, 4), (4, 7), (7, 10),
        
        # Right leg
        (0, 2), (2, 5), (5, 8), (8, 11),
        
        # Left arm
        (9, 13), (13, 16), (16, 18), (18, 20),
        
        # Right arm
        (9, 14), (14, 17), (17, 19), (19, 21),
        
        # Jaw
        (15, 22),
    ]
    return bones


