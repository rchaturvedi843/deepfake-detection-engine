import cv2
import os

video_folder = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\archive\videos"
output_folder = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\video_frames"

frame_interval = 10

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for video_name in os.listdir(video_folder):

    video_path = os.path.join(video_folder, video_name)

    cap = cv2.VideoCapture(video_path)

    count = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:

            frame_name = f"{video_name}_{saved}.jpg"

            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)

            saved += 1

        count += 1

    cap.release()

print("Frame extraction complete")

