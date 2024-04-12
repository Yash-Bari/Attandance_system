import streamlit as st
import cv2
import face_recognition
import os
from datetime import datetime

# Function to recognize faces in the image
def recognize_faces(known_faces_folder):
    known_face_encodings = []
    known_face_names = []

    # Load known faces
    for file in os.listdir(known_faces_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_folder, file))
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file)[0])

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    attendance_taken = False  # Flag to track if attendance has been taken

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break

        # Convert the image from BGR color to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if any known face matches
            if True in matches and not attendance_taken:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Log attendance
                log_attendance(name)
                attendance_taken = True

            # Draw rectangle around the face and write name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or attendance_taken:
            break

    # Release webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    if attendance_taken:
        st.success("Attendance taken successfully.")

# Function to log attendance
def log_attendance(name):
    with open("attendance.csv", "a") as file:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{name},{date_time}\n")

def main():
    st.title("Face Recognition Attendance System")
    st.write("Click the button below to start attendance.")

    if st.button("Take Attendance"):
        known_faces_folder = "known_faces"
        recognize_faces(known_faces_folder)

if __name__ == "__main__":
    main()
