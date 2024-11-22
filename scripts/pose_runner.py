import os
import subprocess


def set_pictures_dir(input_path, output_path, seg_path=None):
    """
    Overwrites the Input and Output lines in the bash script
    """
    if seg_path:
        script_file = depth_script
    with open(script_file) as f:
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

    print(f"overwriting {script_file}")
    print(f"seg path: {seg_path}")
    with open(script_file, 'w') as f:
        f.writelines(lines)


def process_pictures():
    """
    Wrapper for running the sapiens demo scripts
    """
    sessions = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
    cameras = ["50591643", "58860488", "60457274", "65906101"]

    sessions = ["s03"]
    cameras = ["50591643"]

    for session in sessions:
        for camera in cameras:
            # train_dir = "/home/dj/Documents/disc_golf_form_analyzer/fit3d/fit3d_train/train/"
            train_dir = fit_3d_dir
            camera_path = os.path.join(train_dir, session, "pictures", camera)
            video_names = os.listdir(camera_path)
            for i, video_name in enumerate(video_names):
                # module level pose flag
                if pose:
                    run_type = "pose"
                else:
                    run_type = "seg"
                task_dir = os.path.join(train_dir, session, run_type)
                depth_dir = os.path.join(train_dir, session, "depth")

                camera_dir = os.path.join(task_dir, camera)
                depth_camera_dir = os.path.join(depth_dir, camera)

                out_path = os.path.join(camera_dir, video_name)
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


                input_dir = os.path.join(camera_path, video_name)

                existing_output = None
                if os.path.exists(os.path.join(out_path, "sapiens_1b")):
                    existing_output = os.path.join(out_path, "sapiens_1b")
                elif os.path.exists(os.path.join(out_path, "sapiens_0.6b")):
                    existing_output = os.path.join(out_path, "sapiens_0.6b")

                if existing_output:
                    output_files = os.listdir(existing_output)
                    frames = os.listdir(input_dir)
                    # pose will have an image and a pose file, seg with have image plus two seg files
                    if len(output_files) == (len(frames) * 2) or len(output_files) == (len(frames) * 3):
                        print(f"{session}/{camera}/{video_name} already processed")
                        if not pose:
                            print("running depth")
                            set_pictures_dir(input_dir, depth_output_dir, existing_output)
                            subprocess.call(depth_script)
                        continue

                # set_pictures_dir(input_dir, out_path)
                # subprocess.call(script_file)

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

    fit_3d_dir = "/media/dj/3CB88F62B88F1992/fit3d"

    process_pictures()