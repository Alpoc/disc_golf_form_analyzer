import os
import subprocess



def set_pictures_dir(input_path, output_path):

    with open(pose_file) as f:
        lines = f.readlines()
        input_line = 0
        for line_number, line in enumerate(lines):
            # print(line)
            if "INPUT=" in line:
                input_line = line_number
                break
        if input_line:
            lines[input_line] = f'INPUT="{input_path}"\n'
            lines[input_line + 1] = f'OUTPUT="{output_path}"\n'
    with open(pose_file, 'w') as f:
        f.writelines(lines)


def process_pictures():
    sessions = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
    cameras = ["50591643", "58860488", "60457274", "65906101"]

    for session in sessions:
        for camera in cameras:
            train_dir = "/home/dj/Documents/disc_golf_form_analyzer/fit3d/fit3d_train/train/"
            camera_path = os.path.join(train_dir, session, "pictures", camera)
            video_names = os.listdir(camera_path)
            for video_name in video_names:
                pose_dir = os.path.join(train_dir, session, "pose")
                camera_dir = os.path.join(pose_dir, camera)
                out_path = os.path.join(camera_dir, video_name)

                if not os.path.exists(pose_dir):
                    os.makedirs(pose_dir)
                if not os.path.exists(camera_dir):
                    os.makedirs(camera_dir)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                input_dir = os.path.join(camera_path, video_name)

                output_1b = os.path.join(out_path, "sapiens_1b")
                if os.path.exists(output_1b):
                    output_files = os.listdir(output_1b)
                    frames = os.listdir(input_dir)

                    if len(output_files) == (len(frames) * 2):
                        print(f"{session}/{camera}/{video_name} already processed")
                        continue

                set_pictures_dir(input_dir, out_path)
                # subprocess.call(pose_file)


if __name__ == "__main__":
    pose_file = "/home/dj/sapiens/lite/scripts/demo/torchscript/overwrite_pose_keypoints133.sh"
    process_pictures()
    print("finished")