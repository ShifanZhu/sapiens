import os
import numpy as np
import cv2

png_path = '/media/hang/8tb-data/datasets/img/20240806_1_250_ai1101_vcam_png.0/20240806_1_250_ai1101_vcam/png/seq_000000/'
mp4_path = '/media/hang/8tb-data/datasets/bedlam2_download/20240806_1_250_ai1101_vcam_mp4/20240806_1_250_ai1101_vcam/mp4/seq_000000.mp4'


def compare_images_to_video(image_folder, video_path, show_diff=True):
    """Compare images in a folder with frames from a video."""
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Found {len(image_files)} images and {total_frames} video frames")
    
    if len(image_files) != total_frames:
        print(f"Warning: Frame count mismatch! Images: {len(image_files)}, Video: {total_frames}")
    
    matches = 0
    differences = []
    
    for idx, img_file in enumerate(image_files):
        img_frame = cv2.imread(os.path.join(image_folder, img_file))
        img_frame = np.transpose(img_frame, (1, 0, 2))
        img_frame = img_frame[:, ::-1]  # Convert HWC to WHC for comparison
        ret, vid_frame = cap.read()
        
        if not ret:
            print(f"Video ended at frame {idx}")
            break
        
        if img_frame.shape != vid_frame.shape:
            print(f"Frame {idx}: Shape mismatch - Image: {img_frame.shape}, Video: {vid_frame.shape}")
            differences.append((idx, img_file, "shape_mismatch"))
            continue
        
        # Compare frames
        diff = cv2.absdiff(img_frame, vid_frame)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        if max_diff == 0:
            matches += 1
        else:
            differences.append((idx, img_file, f"max_diff={max_diff:.2f}, mean={mean_diff:.2f}"))
        
        if show_diff:
            status = "MATCH" if max_diff == 0 else "DIFF"
            color = (0, 255, 0) if max_diff == 0 else (0, 0, 255)
            combined = np.hstack([img_frame, vid_frame, cv2.applyColorMap((diff * 10).astype(np.uint8), cv2.COLORMAP_HOT)])
            cv2.putText(combined, f"Frame {idx} [{status}]: max={max_diff:.1f} mean={mean_diff:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('Image | Video | Diff (10x)', combined)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nResults: {matches}/{len(image_files)} frames match perfectly")
    if differences:
        print(f"\nDifferences found in {len(differences)} frames:")
        for idx, fname, info in differences[:10]:
            print(f"  Frame {idx} ({fname}): {info}")
    
    return matches == len(image_files)


if __name__ == '__main__':
    are_same = compare_images_to_video(png_path, mp4_path, show_diff=True)
    print(f"\nImages and video are {'IDENTICAL' if are_same else 'DIFFERENT'}")