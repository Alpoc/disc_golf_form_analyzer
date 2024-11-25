import os
import subprocess


def set_pictures_dir(input_path, output_path, seg_path=None):
    """
    Overwrites the Input and Output lines in the bash script
    """
    bash_script = script_file
    if seg_path:
        bash_script = depth_script
    with open(bash_script) as f:
        lines = f.readlines()
        input_line = 0
        for line_number, line in enumerate(lines):
            if "INPUT=" in line:
                input_line = line_number
                break
        if input_line:
            lines[input_line] = f'INPUT="{input_path}"\n'
            lines[input_line + 1] = f'OUTPUT="{output_path}"\n'
            if seg_path:
                lines[input_line + 2] = f'SEG_DIR="{seg_path}"\n'

    with open(bash_script, 'w') as f:
        f.writelines(lines)


def check_for_previous_run(picture_dir, task_dir):
    """
    pictures: list of picture names
    task_dir: output dir from previous runs
    """
    existing_path = None
    # Descend through and use the best data.
    for model_size in ["sapiens_1b", "sapiens_0.6b", "sapiens_0.3b"]:
        if os.path.exists(os.path.join(task_dir, model_size)):
            existing_path = os.path.join(task_dir, model_size)
        if existing_path:
            # Sapiens outputs the annotated images with the same name as input. probably
            last_picture = os.listdir(picture_dir)
            last_picture.sort()
            last_picture = last_picture[-1]
            existing_outputs = os.listdir(existing_path)
            if last_picture in existing_outputs:
                return existing_path

    return None


def run_depth(picture_dir, seg_dir, depth_dir):
    """
    picture_dir: path to input pictures
    seg_dir: path to possibly empty seg dir
    depth_dir: depth dir containing or not containing depth
    """
    existing_depth = check_for_previous_run(picture_dir, depth_dir)
    if existing_depth:
        print(f"Already processed depth for {depth_dir}")
    else:
        existing_seg = check_for_previous_run(picture_dir, seg_dir)
        if not existing_seg:
            print("Running segmentation")
            set_pictures_dir(picture_dir, seg_dir)
            subprocess.call(script_file)
            # Hacky way to get seg dir plus model folder
            existing_seg = check_for_previous_run(picture_dir, seg_dir)
        if existing_seg:
            print(f"Running Depth from {existing_seg}")
            set_pictures_dir(picture_dir, depth_dir, existing_seg)
            subprocess.call(script_file)
        else:
            print(f"Could not find seg at {existing_seg}")
    exit()

def process_pictures():
    """
    Wrapper for running the sapiens demo scripts
    """
    sessions = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
    cameras = ["50591643", "58860488", "60457274", "65906101"]

    sessions = ["s03"]
    cameras = ["50591643"]
    sessions = ["s02"]
    cameras = ["no_camera_angles"]

    for session in sessions:
        for camera in cameras:
            train_dir = fit_3d_dir
            camera_path = os.path.join(train_dir, session, "pictures", camera)
            video_names = os.listdir(camera_path)
            for i, video_name in enumerate(video_names):
                # module level pose flag
                if pose:
                    run_type = "pose"
                else:
                    run_type = "seg"
                # fit3d/03/seg or  fit3d/03/pose
                task_dir = os.path.join(train_dir, session, run_type)
                # fit3d/03/depth
                depth_dir = os.path.join(train_dir, session, "depth")
                # fit3d/03/seg/50591634
                camera_dir = os.path.join(task_dir, camera)
                # fit3d/03/depth/50591634
                depth_camera_dir = os.path.join(depth_dir, camera)

                # fit3d/03/seg/50591634/deadlift, or pose...
                out_path = os.path.join(camera_dir, video_name)
                # fit3d/03/depth/50591634/deadlift
                depth_output_dir = os.path.join(depth_camera_dir, video_name)


                if not os.path.exists(task_dir):
                    os.makedirs(task_dir)
                if not os.path.exists(camera_dir):
                    os.makedirs(camera_dir)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                if not os.path.exists(depth_camera_dir):
                    os.makedirs(depth_camera_dir)
                if not os.path.exists(depth_dir):
                    os.makedirs(depth_dir)
                if not os.path.exists(depth_output_dir):
                    os.makedirs(depth_output_dir)


                input_pictures_dir = os.path.join(camera_path, video_name)

                if not pose:
                    run_depth(input_pictures_dir, out_path, depth_output_dir)
                else:
                    if check_for_previous_run(input_pictures_dir, out_path):
                        print(f"{session}/{camera}/{video_name} already processed")
                        continue
                    else:
                        set_pictures_dir(input_pictures_dir, out_path)
                        subprocess.call(script_file)

                print(f"Processed {i}/{len(video_names)} videos")


if __name__ == "__main__":
    # False == segmentation
    pose = False

    script_dir = "/home/dj/sapiens/lite/scripts/demo/torchscript"

    if pose:
        script_file = os.path.join("overwrite_pose_keypoints133.sh")
    else:
        script_file = os.path.join(script_dir, "overwrite_seg.sh")

    depth_script = os.path.join(script_dir, "overwrite_depth.sh")

    fit_3d_dir = "/media/dj/3CB88F62B88F1992/fit3d/test"

    process_pictures()