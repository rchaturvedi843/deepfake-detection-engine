import cv2
import os

# ==============================
# DATASET PATH
# ==============================

dataset_path = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\archive (1)\FaceForensics++_C23"

output_path = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\ff_frames"

frame_interval = 10


def extract_frames(video_folder, label):

    save_folder = os.path.join(output_path, label)

    os.makedirs(save_folder, exist_ok=True)

    for video in os.listdir(video_folder):

        video_path = os.path.join(video_folder, video)

        if not video_path.endswith(".mp4"):
            continue

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:

                frame_name = f"{video}_{saved_count}.jpg"

                cv2.imwrite(
                    os.path.join(save_folder, frame_name),
                    frame
                )

                saved_count += 1

            frame_count += 1

        cap.release()

        print("Processed:", video)


# ==============================
# PROCESS DATASET
# ==============================

extract_frames(os.path.join(dataset_path,"original"),"Real")

fake_folders = [
"Deepfakes",
"Face2Face",
"FaceSwap",
"FaceShifter",
"NeuralTextures",
"DeepFakeDetection"
]

for folder in fake_folders:

    extract_frames(
        os.path.join(dataset_path,folder),
        "Fake"
    )


print("\nFrame extraction complete.")

