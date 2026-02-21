import cv2
import os

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Ask for user ID or name
name = input("Enter your ID:")

# Create dataset folder
dataset_path = "dataset/" + name
os.makedirs(dataset_path, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        count += 1

        # Crop face
        face_img = gray[y:y+h, x:x+w]

        # Save face image
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face_img)

        # Draw green rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture Face", frame)

    # Stop after 30 images or press Q
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Images saved: {count}")
print(f"Saved in folder: {dataset_path}")