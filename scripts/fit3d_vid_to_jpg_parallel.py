import cv2
import os
import multiprocessing

def convert_vid_to_jpg(video_path, out_path):
    video = cv2.VideoCapture(video_path)
    count = 0
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            image_name = os.path.join(out_path, str(count).zfill(4) + ".jpg")
            cv2.imwrite(image_name, frame)
            count += 1
            # print(f"processed {count}/{num_frames} images")
        else:
            break

    video.release()
    cv2.destroyAllWindows()

    print("output images saved to ", out_path)


def loop_through_cameras(session):
    # cameras = ["50591643", "58860488", "60457274", "65906101"]
    cameras = ["no_camera_angles"]
    for camera in cameras:
        train_dir = "/media/dj/3CB88F62B88F1992/fit3d/test/"
        camera_path = os.path.join(train_dir, session, "videos", camera)
        video_names = os.listdir(camera_path)
        for video_name in video_names:
            # convert_vid_to_jpg(video_name)
            video_path = os.path.join(camera_path, video_name)
            # print(video_path)

            # out_path = os.path.join(camera_path, "pictures", video_name.split(".")[0])
            picture_dir = os.path.join(train_dir, session, "pictures")
            camera_dir = os.path.join(picture_dir, camera)
            out_path = os.path.join(camera_dir, video_name.split(".")[0])

            if not os.path.exists(picture_dir):
                os.makedirs(picture_dir)
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            convert_vid_to_jpg(video_path, out_path)

if __name__ == "__main__":
    # sessions = ["s04", "s05", "s07", "s08", "s09", "s10", "s11"]
    sessions = ["s02", "s12", "s13"]
    pool = multiprocessing.Pool()
    pool.map(loop_through_cameras, sessions)
    print("Finished processing images")
