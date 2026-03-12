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

import torch
import smplx
import trimesh





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter



def visualize_joints_with_video(
    world_joint_frame: np.ndarray,
    cam_joint_frame: np.ndarray,
    video_frames_1: np.ndarray,
    video_frames_2: np.ndarray | None = None,
    bones: list[tuple[int, int]] | None = None,
    stride: int = 1,
    save_path: str | None = None,
    fps: int = 25,
    view_elev: float = 0.0,
    view_azim: float = -180.0,
    axis_range: float = 2.0,
    world_vertices_frame: np.ndarray | None = None,  # [n_frames, n_verts, 3]
    cam_vertices_frame: np.ndarray | None = None,    # [n_frames, n_verts, 3]
    mesh_faces: np.ndarray | None = None,            # [n_faces, 3]
    mesh_alpha: float = 0.15,
    vertex_size: float = 1.0,
):
    """
    2x3 layout:
      Row 1: video1 | joints_world | joints_cam
      Row 2: video2 | vertices_world | vertices_cam
    """
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if world_joint_frame.shape != cam_joint_frame.shape:
        raise ValueError("world_joint_frame and cam_joint_frame must have same shape")
    if world_joint_frame.shape[0] != video_frames_1.shape[0]:
        raise ValueError("video_frames_1 and joints must have same frame count")

    if video_frames_2 is None:
        video_frames_2 = video_frames_1

    if video_frames_2.shape[0] != world_joint_frame.shape[0]:
        raise ValueError("video_frames_2 and joints must have same frame count")

    data_world = world_joint_frame[::stride]
    data_cam = cam_joint_frame[::stride]
    video1 = video_frames_1[::stride]
    video2 = video_frames_2[::stride]
    n_frames = data_world.shape[0]

    if world_vertices_frame is not None:
        world_vertices_frame = world_vertices_frame[::stride]
        if world_vertices_frame.shape[0] != n_frames:
            raise ValueError("world_vertices_frame frame count mismatch")
    if cam_vertices_frame is not None:
        cam_vertices_frame = cam_vertices_frame[::stride]
        if cam_vertices_frame.shape[0] != n_frames:
            raise ValueError("cam_vertices_frame frame count mismatch")

    if (world_vertices_frame is not None or cam_vertices_frame is not None) and mesh_faces is None:
        # Allowed: fallback to scatter rendering
        pass

    fig = plt.figure(figsize=(18, 9))

    # Row 1
    ax_img1 = fig.add_subplot(2, 3, 1)
    ax_img1.axis("off")
    im1 = ax_img1.imshow(video1[0])
    ax_img1.set_title("Video 1")

    ax_jw = fig.add_subplot(2, 3, 2, projection="3d")
    ax_jc = fig.add_subplot(2, 3, 3, projection="3d")

    # Row 2
    ax_img2 = fig.add_subplot(2, 3, 4)
    ax_img2.axis("off")
    im2 = ax_img2.imshow(video2[0])
    ax_img2.set_title("Video 2")

    ax_vw = fig.add_subplot(2, 3, 5, projection="3d")
    ax_vc = fig.add_subplot(2, 3, 6, projection="3d")

    def _setup_3d_axis(ax, title: str):
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-axis_range, axis_range)
        ax.set_ylim(-axis_range, axis_range)
        ax.set_zlim(-axis_range, axis_range)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=view_elev, azim=view_azim)
        # origin
        ax.scatter([0], [0], [0], c="red", marker="x", s=40)

    _setup_3d_axis(ax_jw, "Joints World")
    _setup_3d_axis(ax_jc, "Joints Cam")
    _setup_3d_axis(ax_vw, "Vertices World")
    _setup_3d_axis(ax_vc, "Vertices Cam")

    # Joints scatters
    jw = data_world[0]
    jc = data_cam[0]
    scat_jw = ax_jw.scatter(jw[:, 0], jw[:, 1], jw[:, 2], s=12, c="tab:blue")
    scat_jc = ax_jc.scatter(jc[:, 0], jc[:, 1], jc[:, 2], s=12, c="tab:orange")

    # Joint bones
    bone_lines_jw = []
    bone_lines_jc = []
    if bones:
        for i, j in bones:
            l1, = ax_jw.plot([jw[i, 0], jw[j, 0]], [jw[i, 1], jw[j, 1]], [jw[i, 2], jw[j, 2]], c="k", lw=1.0)
            l2, = ax_jc.plot([jc[i, 0], jc[j, 0]], [jc[i, 1], jc[j, 1]], [jc[i, 2], jc[j, 2]], c="k", lw=1.0)
            bone_lines_jw.append(l1)
            bone_lines_jc.append(l2)

    # Vertices renderers
    vw_artist = None
    vc_artist = None
    vw_scatter = None
    vc_scatter = None

    def _draw_mesh(ax, verts, faces, color):
        return ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces,
            color=color,
            alpha=mesh_alpha,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    if world_vertices_frame is not None:
        if mesh_faces is not None:
            vw_artist = _draw_mesh(ax_vw, world_vertices_frame[0], mesh_faces, "tab:blue")
        else:
            v0 = world_vertices_frame[0]
            vw_scatter = ax_vw.scatter(v0[:, 0], v0[:, 1], v0[:, 2], s=vertex_size, c="tab:blue", alpha=0.7)

    if cam_vertices_frame is not None:
        if mesh_faces is not None:
            vc_artist = _draw_mesh(ax_vc, cam_vertices_frame[0], mesh_faces, "tab:orange")
        else:
            v0 = cam_vertices_frame[0]
            vc_scatter = ax_vc.scatter(v0[:, 0], v0[:, 1], v0[:, 2], s=vertex_size, c="tab:orange", alpha=0.7)

    def update(f):
        nonlocal vw_artist, vc_artist

        # update videos
        im1.set_array(video1[f].astype(np.uint8))
        im2.set_array(video2[f].astype(np.uint8))
        ax_img1.set_title(f"Video 1 | frame {f+1}/{n_frames}")
        ax_img2.set_title(f"Video 2 | frame {f+1}/{n_frames}")

        # update joints
        jwf = data_world[f]
        jcf = data_cam[f]
        scat_jw._offsets3d = (jwf[:, 0], jwf[:, 1], jwf[:, 2])
        scat_jc._offsets3d = (jcf[:, 0], jcf[:, 1], jcf[:, 2])

        if bones:
            for line, (i, j) in zip(bone_lines_jw, bones):
                line.set_data([jwf[i, 0], jwf[j, 0]], [jwf[i, 1], jwf[j, 1]])
                line.set_3d_properties([jwf[i, 2], jwf[j, 2]])
            for line, (i, j) in zip(bone_lines_jc, bones):
                line.set_data([jcf[i, 0], jcf[j, 0]], [jcf[i, 1], jcf[j, 1]])
                line.set_3d_properties([jcf[i, 2], jcf[j, 2]])

        # update vertices world
        if world_vertices_frame is not None:
            vwf = world_vertices_frame[f]
            if mesh_faces is not None:
                if vw_artist is not None:
                    vw_artist.remove()
                vw_artist = _draw_mesh(ax_vw, vwf, mesh_faces, "tab:blue")
            else:
                vw_scatter._offsets3d = (vwf[:, 0], vwf[:, 1], vwf[:, 2])

        # update vertices cam
        if cam_vertices_frame is not None:
            vcf = cam_vertices_frame[f]
            if mesh_faces is not None:
                if vc_artist is not None:
                    vc_artist.remove()
                vc_artist = _draw_mesh(ax_vc, vcf, mesh_faces, "tab:orange")
            else:
                vc_scatter._offsets3d = (vcf[:, 0], vcf[:, 1], vcf[:, 2])

        return [im1, im2, scat_jw, scat_jc] + bone_lines_jw + bone_lines_jc

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False, repeat=False)
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


def cam_to_right_handed(x, y, z, yaw, pitch, roll):
    # Convert from (x=forward, y=left, z=up) to (x=right, y=down, z=forward)
    y = -y
    yaw = np.array(yaw)
    pitch = np.array(pitch)
    roll = np.array(roll)
    yaw = 360-yaw
    pitch = 360-pitch
    return x, y, z, yaw, pitch, roll

def project_mesh_on_video(
    video_frames: np.ndarray,          # [T, H, W, 3], RGB
    vertices_cam: np.ndarray,          # [T, V, 3], (x=forward, y=left, z=up)
    faces: np.ndarray,                 # [F, 3], vertex indices
    K: np.ndarray,                     # [3, 3]
    stride: int = 1,
    mesh_color: tuple[int, int, int] = (0, 255, 255),  # BGR
    alpha: float = 0.35,
    draw_edges: bool = True,
    edge_color: tuple[int, int, int] = (255, 255, 255),  # BGR
    edge_thickness: int = 1,
    save_path: str | None = None,
    fps: int = 6,
) -> np.ndarray:
    """
    Render SMPL-X mesh overlay on video using triangle rasterization (painter's algorithm).
    """
    frames = video_frames[::stride]
    verts_all = vertices_cam[::stride]
    if len(frames) != len(verts_all):
        raise ValueError("video_frames and vertices_cam must have same frame count")

    out_frames = []

    for t in range(len(frames)):
        rgb = frames[t].copy()
        h, w = rgb.shape[:2]

        # OpenCV drawing in BGR
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = base.copy()

        v = verts_all[t]  # (forward,left,up)

        # Convert to OpenCV camera coords: (right,down,forward)
        vcv = np.empty_like(v)
        vcv[:, 0] = -v[:, 1]  # right
        vcv[:, 1] = -v[:, 2]  # down
        vcv[:, 2] =  v[:, 0]  # forward (depth)

        z = vcv[:, 2]
        valid_v = z > 1e-8

        # Project all vertices
        proj = (K @ vcv.T).T
        uv = proj[:, :2] / (proj[:, 2:3] + 1e-8)  # [V,2]
        uv_i = np.round(uv).astype(np.int32)

        # Depth sort triangles (far -> near) for painter's algorithm
        tri_z = z[faces].mean(axis=1)
        order = np.argsort(tri_z)[::-1]  # far first

        for fi in order:
            i0, i1, i2 = faces[fi]
            if not (valid_v[i0] and valid_v[i1] and valid_v[i2]):
                continue

            tri = np.array([uv_i[i0], uv_i[i1], uv_i[i2]], dtype=np.int32)

            # quick bbox check
            xmin, ymin = tri[:, 0].min(), tri[:, 1].min()
            xmax, ymax = tri[:, 0].max(), tri[:, 1].max()
            if xmax < 0 or ymax < 0 or xmin >= w or ymin >= h:
                continue

            cv2.fillConvexPoly(overlay, tri, mesh_color, lineType=cv2.LINE_AA)
            if draw_edges:
                cv2.polylines(
                    overlay, [tri], isClosed=True,
                    color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA
                )

        blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
        out_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        out_frames.append(out_rgb)

    out_frames = np.asarray(out_frames, dtype=np.uint8)

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        hh, ww = out_frames[0].shape[:2]
        writer = cv2.VideoWriter(save_path, fourcc, fps, (ww, hh))
        for fr in out_frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

    return out_frames