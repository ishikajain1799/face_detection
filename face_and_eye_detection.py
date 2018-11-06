
# coding: utf-8

# In[10]:


import numpy as np
import cv2
face_classifier = cv2.CascadeClassifier('C:\\Users\\Admin\\Downloads\\opencv\\sources\\data\\haarcascadeshaarcascade_frontalface_default.xml')

image = cv2.imread('C:\\Users\\Admin\\Desktop\\myimage.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)
eye_classifier = cv2.CascadeClassifier('C:\\Users\\Admin\\Downloads\\opencv\\sources\\data\\haarcascadeshaarcascade_eye.xml')

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (127,0,255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for(ex, ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

