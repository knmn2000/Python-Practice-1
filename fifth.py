\import cv2
import numpy as np
import matplotlib.pyplot as plt


'''cap = cv2.VideoCapture(0)

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
'''

'''
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([87 ,76, 92])
    upper_pink = np.array([255, 255 , 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5,5), np.uint8)
    #erosion = cv2.erode(mask,kernel, iterations = 1)
    #dilation = cv2.dilate(mask, kernel, iterations = 1)

    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel )
    blackhat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel )

    cv2.imshow('res', res)
    cv2.imshow('e', tophat)
    cv2.imshow('d', blackhat)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()'''

'''cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([87, 76, 92])
    upper_pink = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''

'''img_bgr = cv2.imread('templates.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

template = cv2.imread('match.jpg', 0)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)

thres = 0.8

loc = np.where(res >= thres)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt , (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('detected', img_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''img = cv2.imread('santra.jpg')
cv2.imshow('santra', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
mask = np.zeros(img.shape[:2], np.uint8)
print(img.shape[:2])

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = [65, 49, 150 ,150]

cv2.grabCut(img, mask, tuple(rect), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()'''

######## CORNER DETECTION

'''corn = cv2.imread('face1.jpg')

gray = cv2.cvtColor(corn, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray) # FEATURES TO TRACK ACCEPTS ONLY FLOAT VALUES

corners = cv2.goodFeaturesToTrack(gray, 2000, 0.1, 10, useHarrisDetector= True)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(corn, (x,y), 3, 255, -1)

cv2.imshow('corner', corn)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

############## FEATURE EXTRATION

'''import matplotlib.pyplot as plt

img1 = cv2.imread('wallet1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGB)
img2 = cv2.imread('wallet4.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, CrossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key= lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags = 2)
plt.imshow(img3)
plt.show()
'''

'''cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    mask = fgbg.apply(frame)

    cv2.imshow('original',frame)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()'''

face_cascade = cv2.CascadeClassifier('FaceHaar.xml')
eye_cascade = cv2.CascadeClassifier('EyeHaar.xml')
img = cv2.imread('aayush.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + w, x:x + w]
    roi_img = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




