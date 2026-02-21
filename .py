import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# 1️⃣ Load recognizer and labels
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
labels = {v:k for k,v in labels.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# 2️⃣ Video capture
# -----------------------------
cam = cv2.VideoCapture(0)
marked = set()  # To avoid duplicate attendance

# -----------------------------
# 3️⃣ Attendance list
# -----------------------------
attendance_data = []

# -----------------------------
# 4️⃣ Recognition loop
# -----------------------------
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)

        if conf < 60:
            roll = labels[id_]

            if roll not in marked:
                now = datetime.now()
                attendance_data.append([roll, now.date(), now.strftime("%H:%M:%S"), "PRESENT"])
                marked.add(roll)

        else:
            roll = "Unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, roll, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5️⃣ Cleanup
# -----------------------------
cam.release()
cv2.destroyAllWindows()

# -----------------------------
# 6️⃣ Save attendance to Excel
# -----------------------------
if len(attendance_data) > 0:
    folder = r"C:\FaceRecognition\Attendance"
    if not os.path.exists(folder):
        os.makedirs(folder)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    file_path = f"{folder}\\Attendance_{date_str}.xlsx"

    df = pd.DataFrame(attendance_data, columns=["Roll No","Date","Time","Status"])

    # If Excel already exists, append
    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(file_path, index=False)

    # Open Excel automatically
    os.startfile(file_path)
    print(f"✅ Attendance saved and opened: {file_path}")
else:
    print("⚠️ No attendance marked.")