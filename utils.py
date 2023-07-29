import cv2

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    try:

        img = cv2.imread(image_path)
        print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('*****Gray***')
        print(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print('***faces****')
        print(faces)
        for (x,y,w,h) in faces:
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            # if len(eyes) >= 2:
            #     return roi_color

        return roi_color

    except Exception as e:
        print(e)


