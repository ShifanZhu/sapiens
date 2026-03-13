"""SMPL-X joint constants for BEDLAM2."""

# Full raw joint count in the BEDLAM2 label files
_NUM_JOINTS_RAW = 127
PELVIS_IDX = 0

JOINT_NAMES = [
    # 0-21: Core body
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    # 22-24: Jaw and eyes (jaw=22 is excluded from active set)
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    # 25-39: Left hand
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    # 40-54: Right hand
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
    # 55-126: Surface landmarks
    "nose", "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
    "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky",
    "right_eye_brow1", "right_eye_brow2", "right_eye_brow3", "right_eye_brow4", "right_eye_brow5",
    "left_eye_brow5", "left_eye_brow4", "left_eye_brow3", "left_eye_brow2", "left_eye_brow1",
    "nose1", "nose2", "nose3", "nose4",
    "right_nose_2", "right_nose_1", "nose_middle", "left_nose_1", "left_nose_2",
    "right_eye1", "right_eye2", "right_eye3", "right_eye4", "right_eye5", "right_eye6",
    "left_eye4", "left_eye3", "left_eye2", "left_eye1", "left_eye6", "left_eye5",
    "right_mouth_1", "right_mouth_2", "right_mouth_3", "mouth_top",
    "left_mouth_3", "left_mouth_2", "left_mouth_1",
    "left_mouth_5", "left_mouth_4", "mouth_bottom", "right_mouth_4", "right_mouth_5",
    "right_lip_1", "right_lip_2", "lip_top", "left_lip_2", "left_lip_1",
    "left_lip_3", "lip_bottom", "right_lip_3",
    "right_contour_1", "right_contour_2", "right_contour_3", "right_contour_4",
    "right_contour_5", "right_contour_6", "right_contour_7", "right_contour_8",
    "contour_middle",
    "left_contour_8", "left_contour_7", "left_contour_6", "left_contour_5",
    "left_contour_4", "left_contour_3", "left_contour_2", "left_contour_1",
]

# Active joint subset: body (0-21) + eyes (23-24, jaw=22 excluded) +
# hands (25-54) + non-face surface landmarks (60-75: toes, heels, fingertips).
# Excludes: jaw=22, nose/eye/ear surface=55-59, dense face mesh=76-126.
ACTIVE_JOINT_INDICES = (
    list(range(0, 22))    # body (pelvis → right_wrist)
    + [23, 24]            # eyes (left_eye_smplhf, right_eye_smplhf)
    + list(range(25, 55)) # hands (left + right, 30 joints)
    + list(range(60, 76)) # non-face surface (toes, heels, fingertips)
)

NUM_JOINTS = len(ACTIVE_JOINT_INDICES)  # 70

# Map original index → new index (-1 if the joint is excluded)
_ORIG_TO_NEW = {orig: new for new, orig in enumerate(ACTIVE_JOINT_INDICES)}

# Kinematic skeleton for visualization, remapped to active joint indices.
# Bones whose endpoints are not both active are automatically dropped.
_SMPLX_BONES_RAW = (
    # Spine and head
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
    # Eyes (jaw=22 removed)
    (15, 23), (15, 24),
    # Left hand
    (20, 25), (25, 26), (26, 27),
    (20, 28), (28, 29), (29, 30),
    (20, 31), (31, 32), (32, 33),
    (20, 34), (34, 35), (35, 36),
    (20, 37), (37, 38), (38, 39),
    # Right hand
    (21, 40), (40, 41), (41, 42),
    (21, 43), (43, 44), (44, 45),
    (21, 46), (46, 47), (47, 48),
    (21, 49), (49, 50), (50, 51),
    (21, 52), (52, 53), (53, 54),
)
SMPLX_SKELETON = tuple(
    (_ORIG_TO_NEW[a], _ORIG_TO_NEW[b])
    for a, b in _SMPLX_BONES_RAW
    if a in _ORIG_TO_NEW and b in _ORIG_TO_NEW
)

# Left-right joint pairs for horizontal flip augmentation (body + hands)
FLIP_PAIRS = (
    (1, 2),   # left_hip <-> right_hip
    (4, 5),   # left_knee <-> right_knee
    (7, 8),   # left_ankle <-> right_ankle
    (10, 11), # left_foot <-> right_foot
    (13, 14), # left_collar <-> right_collar
    (16, 17), # left_shoulder <-> right_shoulder
    (18, 19), # left_elbow <-> right_elbow
    (20, 21), # left_wrist <-> right_wrist
    (23, 24), # left_eye_smplhf <-> right_eye_smplhf
    # Left hand <-> Right hand (25-39 <-> 40-54)
    (25, 40), (26, 41), (27, 42),
    (28, 43), (29, 44), (30, 45),
    (31, 46), (32, 47), (33, 48),
    (34, 49), (35, 50), (36, 51),
    (37, 52), (38, 53), (39, 54),
)

# ImageNet normalization constants (used for RGB)
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD  = (0.229, 0.224, 0.225)

# Depth normalization: clip to this max distance (meters) then divide
DEPTH_MAX_METERS = 20.0

# Video source FPS (BEDLAM2 is 30fps, downsampled to 6fps)
SOURCE_FPS = 30
TARGET_FPS = 6
FRAME_STRIDE = SOURCE_FPS // TARGET_FPS  # = 5
