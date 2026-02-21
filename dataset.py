import cv2
import os

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv2.VideoCapture(0)

# Ask for user name / ID
user_name = input("Enter your name or ID: ")

# Create folder for this user inside dataset
user_folder = f"dataset/{user_name}"
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

count = 0  # Counter for saved images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the face image
        face_img = gray[y:y + h, x:x + w]
        count += 1
        cv2.imwrite(f"{user_folder}/{user_name}_{count}.jpg", face_img)

    cv2.imshow("Face Detection & Save", frame)

    # Stop after 50 images or press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to folder: {user_folder}")