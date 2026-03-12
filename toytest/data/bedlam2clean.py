import numpy as np
import os
import tarfile
import shutil
import time


def extract_tar(tar_path, dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.dirname(os.path.abspath(tar_path))

    os.makedirs(dest_dir, exist_ok=True)
    log_message(f"Extracting {tar_path} to {dest_dir}...")
    with tarfile.open(tar_path, mode="r:*") as tar:
        tar.extractall(path=dest_dir)
    log_message(f"Extraction completed: {tar_path} to {dest_dir}")
    return dest_dir


def is_tar_complete(tar_path, timeout=5):
    """Check if tar file is complete and not being written to."""
    if not os.path.exists(tar_path):
        return False
    
    try:
        # Check if file size is stable (not changing)
        size1 = os.path.getsize(tar_path)
        time.sleep(timeout)
        size2 = os.path.getsize(tar_path)
        
        if size1 != size2:
            log_message(f"{os.path.basename(tar_path)} size changing: {size1} -> {size2}, still downloading...")
            return False  # Still downloading
        
        # Quick validation: just try to open tar header (no full read)
        with tarfile.open(tar_path, mode="r:*") as tar:
            tar.next()  # Fast: only reads tar header, not entire file
        
        return True
    except (tarfile.TarError, EOFError):
        log_message(f"Tar file {tar_path} is not a valid tar or is incomplete.")
        return False  # Corrupted or incomplete tar

def log_message(message):
    global log_path
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with open(log_path, 'a') as f:
        f.write(message + '\n')
    print(message)

def wait_for_tar(tar_path, check_interval=10, max_wait=3600):
    """Wait until tar file download completes."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if is_tar_complete(tar_path):
            log_message(f"Tar file ready: {tar_path}")
            return True
        log_message(f"Tar still downloading... waiting {check_interval}s")
        time.sleep(check_interval)
    
    log_message(f"Timeout waiting for tar: {tar_path}")
    return False


def clean(src_path):
    while True:
        files = os.listdir(src_path)
        if files[0][:3] == 'seq':
            break
        src_path = os.path.join(src_path, files[0])
    folders = os.listdir(src_path)
    folders.sort()
    for folder in folders:
        folder_path = os.path.join(src_path, folder)
        files = os.listdir(folder_path)
        for f in files:
            idx = int(f.split('.exr')[0][-4:])
            if idx % 5 != 0:
                os.remove(os.path.join(folder_path, f))
    log_message(f"Cleaning completed in {src_path}")
        

def clean_bedlam_data(bedlam_data_dir):
    finished_pth = str(os.path.join(bedlam_data_dir, 'finished.txt'))
    started_pth = str(os.path.join(bedlam_data_dir, 'started.txt'))
    if not os.path.exists(finished_pth):
        open(finished_pth, 'w').close()
        finished = []
    else:
        with open(finished_pth, 'r') as f:
            finished = [line.strip() for line in f.readlines()]
    if not os.path.exists(started_pth):
        open(started_pth, 'w').close()
        started = []
    else:
        with open(started_pth, 'r') as f:
            started = [line.strip() for line in f.readlines()]
    files = os.listdir(bedlam_data_dir)
    files = [f for f in files if f.endswith('.tar')]
    files.sort()  # Sort files to process in order
    log_message(f"Found {files} in {bedlam_data_dir}, start processing...")
    for file in files:
        project_name = file.split('.tar')[0]
        if project_name in finished:
            log_message(f"Project {project_name} already finished, skipping {file}")
            continue
        log_message(f"Processing project {project_name} with file {file}...")
        if project_name not in started:
            started.append(project_name)
            with open(started_pth, 'a') as f:
                f.write(project_name + '\n')
        tar_path = os.path.join(bedlam_data_dir, file)
        tar_completed = is_tar_complete(tar_path)
        if not tar_completed:
            continue
        else:
            log_message(f"Tar file {tar_path} is complete, start processing...")
        extract_tar(tar_path, os.path.join(bedlam_data_dir, project_name))
        os.remove(tar_path)
        log_message(f"Extraction completed and tar removed for {project_name}, start cleaning...")
        clean(os.path.join(bedlam_data_dir, project_name))
        with open(finished_pth, 'a') as f:
            f.write(project_name + '\n')
        log_message(f"Project {project_name} finished, cleaned {file}")

        


    

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Clean Bedlam2 data by extracting tar files and removing unnecessary frames.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing Bedlam2 tar files.')
    args = parser.parse_args()
    bedlam_data_dir = args.data_dir

    # tar_path = os.path.join(bedlam_data_dir, '20240806_1_250_ai1101_vcam_exr_depth.1.tar')
    # extract_tar(tar_path)
    global log_path
    log_path = os.path.join(bedlam_data_dir, 'log.txt')
    while True:
        clean_bedlam_data(bedlam_data_dir)
        time.sleep(1200)  # Check every 60 seconds for new tar files to process
    #print(is_tar_complete('/media/hang/8tb-data/datasets/depth1/20240808_1_250_ai1105_vcam_exr_depth.6.tar'))