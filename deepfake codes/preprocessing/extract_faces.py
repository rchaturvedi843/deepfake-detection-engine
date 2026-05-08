import cv2
import os

input_path = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\ff_frames"
output_path = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\ff_faces"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(label):

    src = os.path.join(input_path, label)
    dst = os.path.join(output_path, label)

    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)

    print(f"\nProcessing {label} images...")
    print("Total images:", len(images))

    processed = 0
    saved = 0

    for img in images:

        img_path = os.path.join(src, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = image[y:y+h, x:x+w]

            face = cv2.resize(face, (224,224))

            save_path = os.path.join(dst, img)

            cv2.imwrite(save_path, face)

            saved += 1

        processed += 1

        if processed % 500 == 0:
            print(f"Processed {processed} images | Faces saved: {saved}")

    print(f"\nFinished {label}")
    print("Faces extracted:", saved)


for label in ["Real", "Fake"]:
    extract_faces(label)

print("\nFace extraction complete.")

