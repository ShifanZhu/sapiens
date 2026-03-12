import numpy as np
import cv2

def play_joints(joints, frame_interval=None, repeat=True, joint_names=None, video_frames=None):

    """
    Play animation of joints in real-time using OpenCV with 3D perspective projection.
    Mouse and keyboard controls for rotation.
    
    Args:
        joints: NumPy array of shape (nframe, njoint, 3) with skeleton coordinates
        frame_interval: Time between frames in seconds (default: 1/30)
        repeat: Whether to loop the animation (default: True)
        joint_names: Optional list of joint names. If None, joints will be named 'joint_0', 'joint_1', etc.
        video_frames: Optional video frames as NumPy array (nframe, H, W, C) to display alongside skeleton
    """
    import time
    
    # Convert numpy array (nframe, njoint, 3) to the expected format
    skel_array = np.array(joints)  # Shape: (nframe, njoint, 3)
    nframe, njoint, _ = skel_array.shape
    
    # Create default joint names if not provided
    if joint_names is None:
        joint_names = [f'joint_{i}' for i in range(njoint)]
    
    # Build head structure
    head = []
    for i in range(njoint):
        head.append({
            'name': joint_names[i]
        })
    
    # Convert frames to expected format
    frames = []
    for frame_idx in range(nframe):
        frame = []
        for joint_idx in range(njoint):
            frame.append({
                'pos': skel_array[frame_idx, joint_idx].tolist()
            })
        frames.append(frame)
    
    if frame_interval is None:
        print("Frame interval not specified, defaulting to 1/30 seconds.")
        frame_interval = 1.0 / 30.0

    fps = 1.0 / frame_interval

    # Compute global range for consistent scaling
    all_positions = []
    for frame in frames:
        for joint_data in frame:
            all_positions.append(joint_data['pos'])
    all_pos_arr = np.array(all_positions)
    
    x_min, x_max = all_pos_arr[:, 0].min(), all_pos_arr[:, 0].max()
    y_min, y_max = all_pos_arr[:, 1].min(), all_pos_arr[:, 1].max()
    z_min, z_max = all_pos_arr[:, 2].min(), all_pos_arr[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    
    # Mouse state
    mouse_state = {'x': 0, 'y': 0, 'pressed': False, 'click_pos': None, 'just_released': False, 'middle_pressed': False}
    selected_joint = {'name': None}  # Track selected joint for name display
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state['pressed'] = True
            mouse_state['x'] = x
            mouse_state['y'] = y
            mouse_state['click_pos'] = (x, y)  # Record click position
            mouse_state['just_released'] = False
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state['pressed'] = False
            # Check if this was a click (small movement)
            if mouse_state['click_pos'] is not None:
                dx = x - mouse_state['click_pos'][0]
                dy = y - mouse_state['click_pos'][1]
                if abs(dx) < 5 and abs(dy) < 5:  # Threshold for click vs drag
                    mouse_state['just_released'] = True  # Mark that we just released from a click
        elif event == cv2.EVENT_MBUTTONDOWN:
            mouse_state['middle_pressed'] = True
            mouse_state['x'] = x
            mouse_state['y'] = y
        elif event == cv2.EVENT_MBUTTONUP:
            mouse_state['middle_pressed'] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_state['pressed']:
                dx = x - mouse_state['x']
                dy = y - mouse_state['y']
                mouse_state['x'] = x
                mouse_state['y'] = y
                
                # Rotation with left click (only if mouse moved significantly)
                if abs(dx) > 1 or abs(dy) > 1:
                    param['angles']['angle_y'] -= dx * 0.5
                    param['angles']['angle_x'] -= dy * 0.5
                    param['angles']['angle_x'] = max(-90, min(90, param['angles']['angle_x']))
                    param['angles']['angle_y'] = param['angles']['angle_y'] % 360
            elif mouse_state['middle_pressed']:
                dx = x - mouse_state['x']
                dy = y - mouse_state['y']
                mouse_state['x'] = x
                mouse_state['y'] = y
                
                # Pan camera with middle click (move in screen space)
                # Convert screen movement to world space movement
                param['camera_offset']['x'] -= dx * 0.1
                param['camera_offset']['y'] += dy * 0.1
            else:
                # Just update position when not pressed
                mouse_state['x'] = x
                mouse_state['y'] = y
    
    def project_3d_to_2d(pos, angle_x=20, angle_y=45, width=800, height=600, focal_length=500):
        """3D perspective projection with rotation"""
        # Translate to origin
        x, y, z = pos[0], pos[1], pos[2]
        
        # Rotation matrices
        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)
        
        # Rotate around Y axis
        cos_y, sin_y = np.cos(angle_y_rad), np.sin(angle_y_rad)
        x_rot = x * cos_y + z * sin_y
        z_rot = -x * sin_y + z * cos_y
        
        # Rotate around X axis
        cos_x, sin_x = np.cos(angle_x_rad), np.sin(angle_x_rad)
        y_rot = y * cos_x - z_rot * sin_x
        z_final = y * sin_x + z_rot * cos_x
        
        # Perspective projection
        z_final += max_range * 2  # Move camera away from center
        if z_final <= 0:
            z_final = 0.1
        
        px = int(width / 2 + (x_rot * focal_length) / z_final)
        py = int(height / 2 + (y_rot * focal_length) / z_final)
        
        return (px, py), z_final
    
    width, height = 800, 600

    frame_idx = 0
    angles = {'angle_x': 0, 'angle_y': 0}
    camera_offset = {'x': 0, 'y': 0, 'z': 0}
    
    # Create window and set mouse callback
    cv2.namedWindow('Joint Animation - 3D')
    callback_param = {'angles': angles, 'camera_offset': camera_offset}
    cv2.setMouseCallback('Joint Animation - 3D', mouse_callback, callback_param)
    
    while True:
        if frame_idx >= len(frames):
            if repeat:
                frame_idx = 0
            else:
                break
        
        start_time = time.time()
        
        # Create blank canvas
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        frame = frames[frame_idx]
        positions = {}
        z_values = {}
        
        angle_x = angles['angle_x']
        angle_y = angles['angle_y']
        
        # Apply panning to projection center
        def project_3d_to_2d_with_pan(pos, angle_x, angle_y, cam_offset_x=0, cam_offset_y=0, cam_offset_z=0, width=800, height=600, focal_length=500):
            """3D perspective projection with Z-up coordinate system.
            
            Coordinate system:
                - X: forward/depth (camera looks along +X)
                - Y: right (positive Y goes to the right on screen)
                - Z: up (positive Z goes up on screen)
            
            This matches BEDLAM/Unreal coordinates:
                - X: forward
                - Y: right
                - Z: up
            """
            # Translate to origin and apply camera offset
            x, y, z = pos[0] - cam_offset_x, pos[1] - cam_offset_y, pos[2] - cam_offset_z
            
            # Rotation matrices
            angle_x_rad = np.radians(angle_x)
            angle_y_rad = np.radians(angle_y)
            
            # Rotate around Z axis (vertical)
            cos_z, sin_z = np.cos(angle_y_rad), np.sin(angle_y_rad)
            x_rot = x * cos_z - y * sin_z
            y_rot = x * sin_z + y * cos_z
            
            # Rotate around Y axis (left/right)
            cos_y, sin_y = np.cos(angle_x_rad), np.sin(angle_x_rad)
            z_rot = z * cos_y - x_rot * sin_y
            x_final = z * sin_y + x_rot * cos_y
            
            # Perspective projection
            # X is depth (front/back), Y is right/left on screen, Z is up/down on screen
            x_final += max_range * 2  # Move camera away
            if x_final <= 0:
                x_final = 0.1
            
            px = int(width / 2 + (y_rot * focal_length) / x_final)  # +Y goes right
            py = int(height / 2 - (z_rot * focal_length) / x_final)  # +Z goes up
            
            return (px, py), x_final
        
        # Draw ground plane grid at z=0 (ground level, since Z is up)
        grid_size = max_range * 2
        grid_step = grid_size / 5
        ground_z = 0  # Ground plane at z=0 (ground level)
        
        # Draw grid lines
        for i in range(-5, 6):
            # Lines parallel to X axis (varying X, constant Y and Z=0)
            p1 = (0 - max_range, 0 - max_range + i * grid_step, ground_z)
            p2 = (0 + max_range, 0 - max_range + i * grid_step, ground_z)
            proj1, _ = project_3d_to_2d_with_pan(p1, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
            proj2, _ = project_3d_to_2d_with_pan(p2, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
            cv2.line(canvas, proj1, proj2, (160, 160, 160), 1)  # Grey
            
            # Lines parallel to Y axis (varying Y, constant X and Z=0)
            p3 = (0 - max_range + i * grid_step, 0 - max_range, ground_z)
            p4 = (0 - max_range + i * grid_step, 0 + max_range, ground_z)
            proj3, _ = project_3d_to_2d_with_pan(p3, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
            proj4, _ = project_3d_to_2d_with_pan(p4, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
            cv2.line(canvas, proj3, proj4, (160, 160, 160), 1)  # Grey
        
        # Draw coordinate axes at world origin
        origin = np.array([0, 0, 0])
        axis_length = 1.0
        
        # X axis (red)
        x_end = origin + np.array([axis_length, 0, 0])
        proj_origin, _ = project_3d_to_2d_with_pan(origin, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
        proj_x_end, _ = project_3d_to_2d_with_pan(x_end, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
        cv2.line(canvas, proj_origin, proj_x_end, (0, 0, 255), 2)
        cv2.putText(canvas, 'X', (proj_x_end[0] + 5, proj_x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Y axis (green)
        y_end = origin + np.array([0, axis_length, 0])
        proj_y_end, _ = project_3d_to_2d_with_pan(y_end, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
        cv2.line(canvas, proj_origin, proj_y_end, (0, 255, 0), 2)
        cv2.putText(canvas, 'Y', (proj_y_end[0] + 5, proj_y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Z axis (blue)
        z_end = origin + np.array([0, 0, axis_length])
        proj_z_end, _ = project_3d_to_2d_with_pan(z_end, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
        cv2.line(canvas, proj_origin, proj_z_end, (255, 0, 0), 2)
        cv2.putText(canvas, 'Z', (proj_z_end[0] + 5, proj_z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw origin point
        cv2.circle(canvas, proj_origin, 8, (0, 0, 0), -1)
        cv2.putText(canvas, 'O', (proj_origin[0] - 8, proj_origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Extract positions and project to 2D
        for idx, joint in enumerate(head):
            name = joint['name']
            pos = np.array(frame[idx]['pos'])
            positions[name] = pos
            proj_2d, z_val = project_3d_to_2d_with_pan(pos, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
            z_values[name] = (proj_2d, z_val)
        
        # Draw joints (sorted by depth)
        joints_with_depth = []
        for name, (proj_2d, z_val) in z_values.items():
            joints_with_depth.append((z_val, name, proj_2d))
        
        joints_with_depth.sort()
        for z_val, name, p in joints_with_depth:
            cv2.circle(canvas, p, 6, (0, 0, 255), -1)
            
            # Only show name if this joint is selected
            if selected_joint['name'] == name:
                # Draw white background for text
                font_scale = 0.35
                thickness = 1
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                bg_p1 = (p[0] + 5, p[1] - text_size[1] - 2)
                bg_p2 = (p[0] + text_size[0] + 8, p[1] + 2)
                cv2.rectangle(canvas, bg_p1, bg_p2, (255, 255, 255), -1)
                cv2.putText(canvas, name, (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Check for joint clicks (when button is released from a small movement)
        if mouse_state['just_released'] and mouse_state['click_pos'] is not None:
            click_x, click_y = mouse_state['click_pos']
            for name, (proj_2d, z_val) in z_values.items():
                dx = click_x - proj_2d[0]
                dy = click_y - proj_2d[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 10:  # Click radius of 10 pixels
                    selected_joint['name'] = name
                    break
            mouse_state['just_released'] = False
            mouse_state['click_pos'] = None
        
        # Add frame counter and controls
        cv2.putText(canvas, f'Frame: {frame_idx}/{len(frames)-1} | FPS: {fps:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f'Angles X:{angle_x:.1f} Y:{angle_y:.1f}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(canvas, 'L-click:rotate | M-click:pan | WASD:move | f:focus | q:quit space:pause', 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        
        # Display with optional video
        if video_frames is not None:
            # Get current video frame and resize to match skeleton canvas height
            vid_frame = video_frames[frame_idx % len(video_frames)]
            vid_h, vid_w = vid_frame.shape[:2]
            scale = height / vid_h
            new_vid_w = int(vid_w * scale)
            vid_resized = cv2.resize(vid_frame, (new_vid_w, height))
            
            # Combine side by side
            combined = np.hstack([vid_resized, canvas])
            cv2.imshow('Video + Skeleton', combined)
        else:
            cv2.imshow('Joint Animation - 3D', canvas)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Pause mode - re-render with current camera settings
            paused = True
            while paused:
                pause_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
                angle_x = angles['angle_x']
                angle_y = angles['angle_y']
                pause_frame = frames[frame_idx]
                
                # Draw grid at z=0 (ground level)
                ground_z = 0
                grid_step_pause = max_range * 2 / 5
                for i in range(-5, 6):
                    # Lines parallel to X axis
                    p1 = (0 - max_range, 0 - max_range + i * grid_step_pause, ground_z)
                    p2 = (0 + max_range, 0 - max_range + i * grid_step_pause, ground_z)
                    proj1, _ = project_3d_to_2d_with_pan(p1, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
                    proj2, _ = project_3d_to_2d_with_pan(p2, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
                    cv2.line(pause_canvas, proj1, proj2, (160, 160, 160), 1)  # Grey
                    # Lines parallel to Y axis
                    p3 = (0 - max_range + i * grid_step_pause, 0 - max_range, ground_z)
                    p4 = (0 - max_range + i * grid_step_pause, 0 + max_range, ground_z)
                    proj3, _ = project_3d_to_2d_with_pan(p3, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
                    proj4, _ = project_3d_to_2d_with_pan(p4, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
                    cv2.line(pause_canvas, proj3, proj4, (160, 160, 160), 1)  # Grey
                
                pause_z_values = {}
                for idx, joint in enumerate(head):
                    name = joint['name']
                    pos = np.array(pause_frame[idx]['pos'])
                    proj_2d, z_val = project_3d_to_2d_with_pan(pos, angle_x, angle_y, camera_offset['x'], camera_offset['y'], camera_offset['z'], width, height)
                    pause_z_values[name] = (proj_2d, z_val)
                
                for name, (proj_2d, z_val) in pause_z_values.items():
                    cv2.circle(pause_canvas, proj_2d, 6, (0, 0, 255), -1)
                    if selected_joint['name'] == name:
                        font_scale = 0.35
                        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        bg_p1 = (proj_2d[0] + 5, proj_2d[1] - text_size[1] - 2)
                        bg_p2 = (proj_2d[0] + text_size[0] + 8, proj_2d[1] + 2)
                        cv2.rectangle(pause_canvas, bg_p1, bg_p2, (255, 255, 255), -1)
                        cv2.putText(pause_canvas, name, (proj_2d[0] + 8, proj_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                
                if mouse_state['just_released'] and mouse_state['click_pos'] is not None:
                    click_x, click_y = mouse_state['click_pos']
                    for name, (proj_2d, z_val) in pause_z_values.items():
                        dx = click_x - proj_2d[0]
                        dy = click_y - proj_2d[1]
                        if np.sqrt(dx*dx + dy*dy) < 10:
                            selected_joint['name'] = name
                            break
                    mouse_state['just_released'] = False
                    mouse_state['click_pos'] = None
                
                cv2.putText(pause_canvas, '[PAUSED]', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(pause_canvas, f'X:{angle_x:.1f} Y:{angle_y:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(pause_canvas, 'L-click:rotate | M-click:pan | WASD:move | arrows:rotate | f:focus | space:resume', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
                
                # Display with optional video in pause mode
                if video_frames is not None:
                    vid_frame = video_frames[frame_idx % len(video_frames)]
                    vid_h, vid_w = vid_frame.shape[:2]
                    scale = height / vid_h
                    new_vid_w = int(vid_w * scale)
                    vid_resized = cv2.resize(vid_frame, (new_vid_w, height))
                    combined_pause = np.hstack([vid_resized, pause_canvas])
                    cv2.imshow('Video + Skeleton', combined_pause)
                else:
                    cv2.imshow('Joint Animation - 3D', pause_canvas)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    paused = False
                elif key == ord('w') or key == ord('a') or key == ord('s') or key == ord('d'):
                    move_speed = 0.5
                    move_x, move_y = 0, 0
                    if key == ord('w'): move_x = move_speed
                    elif key == ord('s'): move_x = -move_speed
                    elif key == ord('d'): move_y = move_speed
                    elif key == ord('a'): move_y = -move_speed
                    angle_y_rad = np.radians(angles['angle_y'])
                    angle_x_rad = np.radians(angles['angle_x'])
                    cos_yaw, sin_yaw = np.cos(angle_y_rad), np.sin(angle_y_rad)
                    cos_pitch, sin_pitch = np.cos(angle_x_rad), np.sin(angle_x_rad)
                    x_temp = move_x * cos_yaw + move_y * sin_yaw
                    y_temp = -move_x * sin_yaw + move_y * cos_yaw
                    final_x = x_temp * cos_pitch
                    final_z = x_temp * sin_pitch
                    camera_offset['x'] += final_x
                    camera_offset['y'] += y_temp
                    camera_offset['z'] += final_z
                elif key == 82: angles['angle_x'] = min(80, angles['angle_x'] + 5)
                elif key == 84: angles['angle_x'] = max(-80, angles['angle_x'] - 5)
                elif key == 81: angles['angle_y'] = (angles['angle_y'] - 5) % 360
                elif key == 83: angles['angle_y'] = (angles['angle_y'] + 5) % 360
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    paused = False
                elif key == ord('d') or key == ord('s') or key == ord('a') or key == ord('w'):
                    # Allow movement during pause
                    move_speed = 0.5
                    move_x, move_y, move_z = 0, 0, 0
                    
                    if key == ord('w'):
                        move_x = move_speed
                    elif key == ord('s'):
                        move_x = -move_speed
                    elif key == ord('d'):
                        move_y = move_speed
                    elif key == ord('a'):
                        move_y = -move_speed
                    
                    angle_y_rad = np.radians(angles['angle_y'])
                    angle_x_rad = np.radians(angles['angle_x'])
                    
                    cos_yaw, sin_yaw = np.cos(angle_y_rad), np.sin(angle_y_rad)
                    cos_pitch, sin_pitch = np.cos(angle_x_rad), np.sin(angle_x_rad)
                    
                    x_temp = move_x * cos_yaw + move_y * sin_yaw
                    y_temp = -move_x * sin_yaw + move_y * cos_yaw
                    z_temp = move_z
                    
                    final_x = x_temp * cos_pitch - z_temp * sin_pitch
                    final_z = x_temp * sin_pitch + z_temp * cos_pitch
                    final_y = y_temp
                    
                    camera_offset['x'] += final_x
                    camera_offset['y'] += final_y
                    camera_offset['z'] += final_z
                elif key == 82:  # Up arrow
                    angles['angle_x'] = min(80, angles['angle_x'] + 5)
                elif key == 84:  # Down arrow
                    angles['angle_x'] = max(-80, angles['angle_x'] - 5)
                elif key == 81:  # Left arrow
                    angles['angle_y'] = (angles['angle_y'] - 5) % 360
                elif key == 83:  # Right arrow
                    angles['angle_y'] = (angles['angle_y'] + 5) % 360
                elif key == ord('f'):  # Focus on selected joint
                    if selected_joint['name'] is not None:
                        for idx, joint in enumerate(head):
                            if joint['name'] == selected_joint['name']:
                                joint_pos = pause_frame[idx]['pos']
                                camera_offset['x'] = joint_pos[0]
                                camera_offset['y'] = joint_pos[1]
                                camera_offset['z'] = joint_pos[2]
                                break
            
            if key == ord('q'):
                break
        elif key == ord('d') or key == ord('s') or key == ord('a') or key == ord('w'):
            # Camera-relative movement
            move_speed = 0.5
            
            # Movement vector in camera space (X=depth/forward, Y=left/right, Z=up/down)
            move_x, move_y, move_z = 0, 0, 0
            
            if key == ord('w'):  # Forward (toward camera facing direction)
                move_x = move_speed
            elif key == ord('s'):  # Backward
                move_x = -move_speed
            elif key == ord('d'):  # Right
                move_y = move_speed
            elif key == ord('a'):  # Left
                move_y = -move_speed
            
            # Calculate radians for angles
            angle_y_rad = np.radians(angles['angle_y'])
            angle_x_rad = np.radians(angles['angle_x'])
            
            # Apply inverse rotations to get world-space movement
            # Inverse of the projection transformations
            cos_yaw, sin_yaw = np.cos(angle_y_rad), np.sin(angle_y_rad)
            cos_pitch, sin_pitch = np.cos(angle_x_rad), np.sin(angle_x_rad)
            
            # First apply inverse yaw (rotation around Z by -angle_y)
            x_temp = move_x * cos_yaw + move_y * sin_yaw
            y_temp = -move_x * sin_yaw + move_y * cos_yaw
            z_temp = move_z
            
            # Then apply inverse pitch (rotation around Y by -angle_x)
            final_x = x_temp * cos_pitch - z_temp * sin_pitch
            final_z = x_temp * sin_pitch + z_temp * cos_pitch
            final_y = y_temp

            # Update camera offset
            camera_offset['x'] += final_x
            camera_offset['y'] += final_y
            camera_offset['z'] += final_z
            
        elif key == 82:  # Up arrow
            angles['angle_x'] = min(80, angles['angle_x'] + 5)
        elif key == 84:  # Down arrow
            angles['angle_x'] = max(-80, angles['angle_x'] - 5)
        elif key == 81:  # Left arrow
            angles['angle_y'] = (angles['angle_y'] - 5) % 360
        elif key == 83:  # Right arrow
            angles['angle_y'] = (angles['angle_y'] + 5) % 360
        elif key == ord('f'):  # Focus on selected joint
            if selected_joint['name'] is not None and selected_joint['name'] in positions:
                # Get the 3D position of the selected joint
                joint_pos = positions[selected_joint['name']]
                # Set camera offset to center the joint at world coordinates
                camera_offset['x'] = joint_pos[0]
                camera_offset['y'] = joint_pos[1]
                camera_offset['z'] = joint_pos[2]
        
        # Control framerate
        elapsed = time.time() - start_time
        sleep_time = max(1, int((frame_interval - elapsed) * 1000))
        time.sleep(sleep_time / 1000.0)
        frame_idx += 1
    
    cv2.destroyAllWindows()

