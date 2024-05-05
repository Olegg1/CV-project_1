import mediapipe as mp
import numpy as np
import cv2
from PIL import Image


cap = cv2.VideoCapture(1)
# nose_file = 'png-transparent-red-color-nose-nose-people-color-sticker-thumbnail.png'
nose_file = 'pngwing.com.png'
star_file = 'pngegg (2).png'
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

nose = ''
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    results = mp_face_mesh.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, point in enumerate(face_landmarks.landmark):
                if i == 4:
                    nose = point
                if i == 133:
                    left = point
                if i == 362:
                    right = point
                if i == 263:
                    r_eye = point
                if i == 159:
                    h_left_up = point
                if i == 145:
                    h_left_low = point 
                if i == 33:
                    w_left_eye_left = point
                if i == 362:
                    right_eye_left = point
                if i == 263:
                    right_eye_right = point
                if i == 386:
                    up_right_eye = point
                if i == 374:
                    low_right_eye = point    
                if i == 254:
                    left_face = point
                if i == 454:
                    right_face = point

    h, w, c = frame.shape
    # img_size_n = 50
    # img_size = 20

    left_eye_center = (h_left_low.y - h_left_up.y) // 2 + h_left_up.y
    width_left_eye = 650* (left.x - w_left_eye_left.x)

    right_eye_center = (low_right_eye.y - up_right_eye.y) // 2 + up_right_eye.y
    width_right_eye = 650* (right_eye_right.x - right_eye_left.x)

    nose_d = 800 * (right_face.x - left_face.x)
    
    if nose:
        nx = int(nose.x * w - nose_d / 2)
        ny = int(nose.y * h - nose_d / 2)
        
        im = cv2.imread(nose_file, cv2.IMREAD_UNCHANGED)
        star = cv2.imread(star_file, cv2.IMREAD_UNCHANGED)

        image_nose = cv2.resize(im, (int(nose_d), int(nose_d)), interpolation=cv2.INTER_LANCZOS4)
        image_star = cv2.resize(star, (int(width_left_eye), int(width_left_eye)), interpolation=cv2.INTER_LANCZOS4)
        nose_data = Image.fromarray(cv2.cvtColor(image_nose, cv2.COLOR_RGBA2BGRA))
        star_data = Image.fromarray(cv2.cvtColor(image_star, cv2.COLOR_RGBA2BGRA))

        img = Image.fromarray(frame)
        img.paste(nose_data, (nx, ny), mask=nose_data)
        
        lx = int(h_left_low.x * w - int(width_left_eye) / 2)
        ly = int(left_eye_center * h - int(width_left_eye) / 2)

        rx = int(low_right_eye.x * w - int(width_left_eye) / 2)
        ry = int(right_eye_center * h - int(width_left_eye) / 2)

        img.paste(star_data, (lx, ly), mask=star_data)

        img.paste(star_data, (rx, ry), mask=star_data)
        frame = np.array(img)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('MediaPipe Face Mesh with Red Nose Circle', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()