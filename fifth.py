import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([87 ,76, 92])
    upper_pink = np.array([255, 255 , 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    med = cv2.medianBlur(res, 15)

    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('med', med)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()