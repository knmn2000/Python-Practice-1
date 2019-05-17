


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
cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('FaceHaar.xml')
eye_cascade = cv2.CascadeClassifier('eyehaarwithglass.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + w, x:x + w]
        roi_img = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.release()
cv2.destroyAllWindows()


######################### DIFFERENT PROGRAM , COMMENTING ERROR
img_bgr = cv2.imread('badiimage.jpeg')
temp = cv2.imread('banda.jpeg')
rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

red_low = np.array([0, 200,21])#([1, 201, 130])
red_high = np.array([9,250,255])

corr = cv2.matchTemplate(rgb, temp, cv2.TM_CCOEFF_NORMED)

mask =  cv2.inRange(rgb, red_low, red_high)

cv2.imshow('jj', mask)
cv2.imshow('aa00.', img_bgr)
cv2.imshow('a', corr)'''
'''template = cv2.imread('banda.jpeg', 0)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)

thres = 0.2

loc = np.where(res >= thres)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt , (pt[0] + w, pt[1] + h), (0,0,0), 2)

cv2.imshow('detected', img_bgr)

""" a = https://kinja.com/jack-loftus-old
https://kinja.com/caseychan
https://kinja.com/kylenw
https://kinja.com/liptakaa
https://kinja.com/rosa
https://kinja.com/chengela
https://kinja.com/mattnovak
https://kinja.com/seanhollister
https://kinja.com/sidneyfussell
https://kinja.com/maddiestone
https://kinja.com/ajdellinger
https://kinja.com/andrewcouts
https://kinja.com/sambiddle
https://kinja.com/catiekeck
https://kinja.com/laurendavis
https://kinja.com/kevin-lee-old
https://kinja.com/tjenningsbrown
https://kinja.com/darrenorf
https://kinja.com/alejandroalba
https://kinja.com/chris-mills
https://kinja.com/dan-nosowitz-old
https://kinja.com/fruitsoftheweb
https://kinja.com/ericlimer
https://kinja.com/robschoon
https://kinja.com/davidnield
https://kinja.com/conger
https://kinja.com/kcampbelldollaghan
https://kinja.com/rmisra
https://kinja.com/kyle-vanhemert-old
https://kinja.com/rtgonzalez
https://kinja.com/rhettjonesgizmodo
https://kinja.com/hudsonhongo
https://kinja.com/williamturton
https://kinja.com/john-herrman-old
https://kinja.com/leahbecerra
https://kinja.com/michaelfnunez
https://kinja.com/alexcranz
https://kinja.com/bryanlufkin
https://kinja.com/ace
https://kinja.com/estheringlis-arkell
https://kinja.com/acovert31
https://kinja.com/evepeyser
https://kinja.com/robertsorokanich
https://kinja.com/georgedvorsky
https://kinja.com/carlivelocci
https://kinja.com/jcondliffe
https://kinja.com/sophiekleeman
https://kinja.com/sean-fallon-old
https://kinja.com/knibbs
https://kinja.com/tommckay
https://kinja.com/dellcam
https://kinja.com/libbywatson
https://kinja.com/Mark-Strauss
https://kinja.com/nicolewetsman
https://kinja.com/melaniehannah
https://kinja.com/annaleenewitz
https://kinja.com/ashleyfeinberg"""
'''
'''
import requests
from bs4 import BeautifulSoup
import urllib

b = a.split()

i = 0
lst = []
lst2 = []

while i < 67:
    url = b[i]
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'html.parser')

    for title in soup.findAll('a', {'class': 'js_entry-link'}):
            #lst.append(title.get('href'))
            lst.append(title.text)

            print(lst[i])
            i += 1

i = 0'''
'''
while i < 67:
    url = b[i]
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'html.parser')

    for bio in soup.findAll('div', {'class':'user-details'}):
        lst2.append(bio.text)
        print(lst2[i])
        i += 1
'''

'''i = input()

arr = input().split()
brr = sorted(arr)

a = arr[0:int(((len(arr)/2))) ]
b = arr[int((len(arr)/2)) :int((len(arr)))]


c = sorted(a)
d = sorted(b)


if arr != brr:
    while c != a or d != b:
        a = arr[0:int(((len(a) / 2)))]
        b = arr[int((len(b) / 2)):int((len(b)))]
        c = sorted(a)
        d = sorted(b)

    if len(a) > len(b):
        print(len(a))
    else:
        print(len(b))
else:
    print(len(arr))'''

'''

if brr == arr :
    print(len(arr))
else:
    a = arr[0:int(((len(arr)/2)))+1]
    b = arr[int((len(arr)/2))-1:int((len(arr)))]
    print(a, b)
'''

###################### DOESNT WORK ?!?!!?
'''
import datetime
import pandas_datareader.data as web
#import pandas.io as web
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()

df = web.DataReader("5_Industry_Portfolios", "famafrench", start, end)

df.reset_index(inplace=True)
df.set_index("Date", inplace=True)
df = df.drop("Symbol", axis=1)

print(df.head())

df['High'].plot()
plt.legend()
plt.show()'''



'''import pandas as pd
import numpy as np
web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce Rate':[65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)
print(df.head())

#print(df.set_index('Day'))
#print(df.head())

print(df[['Bounce Rate', 'Day']])
print(df.Visitors.tolist())
print(df['Visitors'].tolist())

########## ^ THIS WORKS WITH A SINGLE COLUMN, BUT FOR MORE THAN 1, YOU NEED TO USE NUMPY ######

print(np.array(df[['Visitors','Bounce Rate']]))

########################### MAGIC - USEFUL FOR CV2 ETC
df2= pd.DataFrame(np.array(df[['Visitors','Bounce Rate']]))
print(df2)
'''
################# HACKERRANK PROBLEM
'''regex_integer_in_range = r"[100000-999999]"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"\{2}"	# Do not delete 'r'.

import re
P = input()

lst = re.findall(r"[1][0,2-9][1]|[2][0-1,3-9][2]|[3][0-2,4-9][3]|[4][0-3,5-9][1]|[5][0-4,6-9][5]|[6][0-5,7-9][6]|[7][0-6,8-9][7]|[1][0-7,9][8]|[9][0-8][9]|", P)
lst2 = []
for i in lst:
    lst2.append(i)

lst3 = [x for x in lst2 if x]

if not lst3:
    print('True')
else:
    print('False')
#print(re.findall(r"[1][0,2-9][1]|[2][0-1,3-9][2]|[3][0-2,4-9][3]|[4][0-3,5-9][1]|[5][0-4,6-9][5]|[6][0-5,7-9][6]|[7][0-6,8-9][7]|[1][0-7,9][8]|[9][0-8][9]|", P))'''

import time
import cv2
from mss import mss
import numpy
from pynput.keyboard import Key, Controller
from directkeys import PressKey, W, A, S, D

sct = mss()

monitor = {"top": 40, "left": 0, "width": 800, "height": 600}
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
PressKey(W)
'''while 1:
        last_time = time.time()

        img = numpy.array(sct.grab(monitor))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        curvy = cv2.Canny(gray_image, 50,150)


        cv2.imshow("Bot Jarodhar (PRE_ALPHA_STAGE)", gray_image)
        cv2.imshow("Bot Jarodhar )", curvy)
        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        print("fps: {}".format(1 / (time.time() - last_time)))

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break'''
