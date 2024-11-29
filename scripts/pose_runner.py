import os
import subprocess
import time


def set_pictures_dir(input_path, output_path, seg_path=None):
    """
    Overwrites the Input and Output lines in the bash script.
    This shouldn't be done and the python file should be used directly but this was simpler to figure out.
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
    print(f"output set to {output_path}")
    with open(bash_script, 'w') as f:
        f.writelines(lines)


def check_for_previous_run(picture_dir, task_dir):
    """
    Takes the list picture in pictures and checks if that picture exists in the task dir.
    Sapiens outputs an annotated picture with the same name, probably.
    pictures: list of picture names
    task_dir: output dir from previous runs
    return: location of already existing dir if the run completed.
    """
    # Descend through and use the best data.
    # print(f"checking for existing data at {task_dir}")
    for model_size in ["sapiens_1b", "sapiens_0.6b", "sapiens_0.3b"]:
        existing_path = os.path.join(task_dir, model_size)
        if os.path.exists(existing_path):
            # TODO: change back to glob for when cleanup doesnt happen...
            last_picture = os.listdir(picture_dir)
            if len(last_picture):
                last_picture.sort()
                last_picture = last_picture[-1]
                existing_outputs = os.listdir(existing_path)
                if last_picture in existing_outputs:
                    return existing_path

    return None


def run_depth(picture_dir, seg_dir, depth_dir):
    """
    Runs seg and depth. If depth exists it returns. If previous seg exists it uses that for depth.
    picture_dir: path to input pictures
    seg_dir: path to possibly empty seg dir
    depth_dir: depth dir containing or not containing depth
    """
    # There is an OOM issue that occasionally occurs. Might have to do with GC not happening fast enough
    time.sleep(1)
    existing_depth = check_for_previous_run(picture_dir, depth_dir)
    if existing_depth:
        print(f"Depth already processed for {depth_dir}")
    else:
        existing_seg = check_for_previous_run(picture_dir, seg_dir)
        if not existing_seg:
            print(f"Running segmentation on {seg_dir}")
            set_pictures_dir(picture_dir, seg_dir)
            subprocess.call(script_file)
            # Hacky way to get seg dir plus model size folder
            # Todo: Bug here still needs to be validated that it's fixed. It was not running depth after seg.
            existing_seg = check_for_previous_run(picture_dir, seg_dir)
            print(f"new seg at {existing_seg}")
            time.sleep(1)
        else:
            print("Previous segmentation found")

        set_pictures_dir(picture_dir, depth_dir, existing_seg)
        print(f"Running: {depth_script}")
        subprocess.call(depth_script)


def process_pictures(sessions, cameras):
    """
    Wrapper for running the sapiens demo scripts
    """

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
                        print(f"Pose already processed for {session}/{camera}/{video_name}")
                        continue
                    else:
                        print("Running pose")
                        set_pictures_dir(input_pictures_dir, out_path)
                        subprocess.call(script_file)

                print(f"Processed {i + 1}/{len(video_names)} videos")


if __name__ == "__main__":
    """
    Put this script inside of `/sapiens/lite/scripts/demo` and run it to generate seg, pose, and depth. 
    This script does not delete the seg files. They take up a lot of space. Remove the overwrite part of the script
    files if you're comfortable with the stock ones being overwritten. Otherwise add `overwrite_` to each script. 
    """
    # False == segmentation and depth
    # Todo: change this so everything is ran.
    pose = False
    script_dir = "/home/dj/sapiens/lite/scripts/demo/torchscript"

    if pose:
        script_file = os.path.join(script_dir, "overwrite_pose_keypoints133.sh")
    else:
        script_file = os.path.join(script_dir, "overwrite_seg.sh")

    depth_script = os.path.join(script_dir, "overwrite_depth.sh")

    fit_3d_dir = os.path.join("/media/dj/3CB88F62B88F1992/fit3d/", "train")
    cameras = ["50591643", "60457274"]
    sessions = ["s03"]
    process_pictures(sessions, cameras)
