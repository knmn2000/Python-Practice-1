# simple hackerrank problem
'''s = list(str(input()))
i = int(input())
r = 0
n = i
for x in range(int(len(s)/i) + int(len(s)%i)):
    print(''.join(s[r:n]))
    r += i
    n += i'''
# fix
'''
i = int(input()) + 1
k = 0
lst = [k for k in range(1,i)]
r = 0
for r in range(i-1):
    l = str(lst[0:r]+lst[r::-1])
    l.strip('[],')
    print(l)'''
'''a = [1, 5, 10, 20, 40, 80]
b = [3, 4, 15, 20, 30, 70, 80, 120]
lst = []
for x in a:
    if x in b:
        lst.append(x)
    else:
        continue
print(lst)'''
'''import random

dataset = ['andra', 'anus', 'arse', 'arsehole', 'ass', 'ass-hat', 'ass-jabber', 'ass-pirate', 'assbag', 'assbandit',
           'assbanger', 'assbite', 'assclown', 'asscock', 'asscracker', 'asses', 'assface', 'assfuck', 'assfucker',
           'assgoblin', 'asshat', 'asshead', 'asshole', 'asshopper', 'assjacker', 'asslick', 'asslicker', 'assmonkey',
           'assmunch', 'assmuncher', 'assnigger', 'asspirate', 'assshit', 'assshole', 'asssucker', 'asswad', 'asswipe',
           'axwound', 'bampot', 'bastard', 'beaner', 'bhosdika', 'bitch', 'bitchass', 'bitches', 'bitchtits', 'bitchy',
           'blow job', 'blowjob', 'bollocks', 'bollox', 'boner', 'brotherfucker', 'bullshit', 'bumblefuck', 'butt plug',
           'butt-pirate', 'buttfucka', 'buttfucker', 'camel toe', 'carpetmuncher', 'chamar', 'champak', 'chesticle',
           'chinc', 'chink', 'choad', 'chode', 'chut', 'chutiya', 'clit', 'clitface', 'clitfuck', 'clusterfuck', 'cock',
           'cockass', 'cockbite', 'cockburger', 'cockface', 'cockfucker', 'cockhead', 'cockjockey', 'cockknoker',
           'cockmaster', 'cockmongler', 'cockmongruel', 'cockmonkey', 'cockmuncher', 'cocknose', 'cocknugget',
           'cockshit', 'cocksmith', 'cocksmoke', 'cocksmoker', 'cocksniffer', 'cocksucker', 'cockwaffle', 'coochie',
           'coochy', 'coon', 'cooter', 'cracker', 'cum', 'cumbubble', 'cumdumpster', 'cumguzzler', 'cumjockey',
           'cumslut', 'cumtart', 'cunnie', 'cunnilingus', 'cunt', 'cuntass', 'cuntface', 'cunthole', 'cuntlicker',
           'cuntrag', 'cuntslut', 'dago', 'damn', 'deggo', 'dick', 'dick-sneeze', 'dickbag', 'dickbeaters', 'dickface',
           'dickfuck', 'dickfucker', 'dickhead', 'dickhole', 'dickjuice', 'dickmilk', 'dickmonger', 'dicks', 'dickslap',
           'dicksucker', 'dicksucking', 'dicktickler', 'dickwad', 'dickweasel', 'dickweed', 'dickwod', 'dike', 'dildo',
           'dipshit', 'doochbag', 'dookie', 'douche', 'douche-fag', 'douchebag', 'douchewaffle', 'dumass', 'dumb ass',
           'dumbass', 'dumbfuck', 'dumbshit', 'dumshit', 'dyke', 'fag', 'fagbag', 'fagfucker', 'faggit', 'faggot',
           'faggotcock', 'fagtard', 'fatass', 'fellatio', 'feltch', 'flamer', 'fuck', 'fuckass', 'fuckbag', 'fuckboy',
           'fuckbrain', 'fuckbutt', 'fuckbutter', 'fucked', 'fucker', 'fuckersucker', 'fuckface', 'fuckhead',
           'fuckhole', 'fuckin', 'fucking', 'fucknut', 'fucknutt', 'fuckoff', 'fucks', 'fuckstick', 'fucktard',
           'fucktart', 'fuckup', 'fuckwad', 'fuckwit', 'fuckwitt', 'fudgepacker', 'gaandu', 'gay', 'gayass', 'gaybob',
           'gaydo', 'gayfuck', 'gayfuckist', 'gaylord', 'gaytard', 'gaywad', 'goddamn', 'goddamnit', 'gooch', 'gook',
           'gringo', 'guido', 'handjob', 'harami', 'hard on', 'heeb', 'hell', 'hoe', 'homo', 'homodumbshit', 'honkey',
           'humping', 'jackass', 'jagoff', 'jap', 'jerk off', 'jerkass', 'jigaboo', 'jizz', 'jungle bunny',
           'junglebunny', 'kike', 'kooch', 'kootch', 'kraut', 'kunt', 'kyke', 'lameass', 'lardass', 'lavde', 'lesbian',
           'lesbo', 'lezzie', 'lodu', 'mcfagget', 'mick', 'minge', 'mothafucka', "mothafuckin\\'", 'motherfucker',
           'motherfucking', 'muff', 'muffdiver', 'munging', 'negro', 'nigaboo', 'nigga', 'nigger', 'niggers', 'niglet',
           'nut sack', 'nutsack', 'paki', 'panooch', 'pecker', 'peckerhead', 'penis', 'penisbanger', 'penisfucker',
           'penispuffer', 'piss', 'pissed', 'pissed off', 'pissflaps', 'piyushXD', 'polesmoker', 'pollock', 'poon',
           'poonani', 'poonany', 'poontang', 'porch monkey', 'porchmonkey', 'prick', 'punanny', 'punta', 'pussies',
           'pussy', 'pussylicking', 'puto', 'queef', 'queer', 'queerbait', 'queerhole', 'raand', 'ramdev', 'randi', 'renob',
           'rimjob', 'ruski', 'sand nigger', 'sandnigger', 'schlong', 'scrote', 'shit', 'shitass', 'shitbag',
           'shitbagger', 'shitbrains', 'shitbreath', 'shitcanned', 'shitcunt', 'shitdick', 'shitface', 'shitfaced',
           'shithead', 'shithole', 'shithouse', 'shitspitter', 'shitstain', 'shitter', 'shittiest', 'shitting',
           'shitty', 'shiz', 'shiznit', 'skank', 'skeet', 'skullfuck', 'slut', 'slutbag', 'smeg', 'snatch', 'spic',
           'spick', 'splooge', 'spook', 'suckass', 'tard', 'tatte', 'tatto ke saudagar', 'testicle', 'thundercunt',
           'tit', 'titfuck', 'tits', 'tittyfuck', 'trivedi', 'twat', 'twatlips', 'twats', 'twatwaffle', 'unclefucker', 'va-j-j',
           'vag', 'vagina', 'vajayjay', 'vjayjay', 'wank', 'wankjob', 'wetback', 'whore', 'whorebag', 'whoreface',
           'wop']


# dataset created by scrapping https://www.noswearing.com/dictionary/

def ListAbusesFrom(data_in):
    _temp = []
    if type(data_in) is not str:
        return ("No Abuse Words from", data_in, "Provide input between a-z instead")
    data_in = str(data_in).lower()
    for i in dataset:
        if i.startswith(data_in):
            _temp.append(i)

    return _temp


def RandomAbuseFrom(data_in):
    _temp = []
    if type(data_in) is not str:
        return ("No Abuse Words from", data_in, "Provide input between a-z instead")
    data_in = str(data_in).lower()
    for i in dataset:
        if i.startswith(data_in):
            _temp.append(i)

    return random.choice(_temp)


def ListAnyAbuse():
    return random.choice(dataset)


def ListAllAbuses():
    return dataset


def ForName(data_in):
    _temp = []
    if type(data_in) is not str:
        return ("No Abuse Words from", data_in, "Provide input between a-z instead")
    data_In = str(data_in[0]).lower()
    for i in dataset:
        if i.startswith(data_In):
            _temp.append(i)
    lst = [data_in + ' ' + x for x in _temp]
    return lst
print(ForName(str(input())))
'''
#hacktoberfest
'''def fibo():
    a = 0
    b = 1
    _temp = [0, 1]
    n = int(input(" n? :  "))

    while (n):
        c = a + b
        a = b
        b = c
        _temp.append(c)
        n = n - 1

    print(_temp[len(_temp) - 3])

fibo()'''



'''

import numpy as np
import random

#a = np.array([[2,4,5],[1,5,3], [5,2,3]])
#a = np.random.random((4,3))
#print(a)
#print("\n")

#print(a[3::1])

#a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
#b = np.array([1,2,2,0])
#print(a[np.arange(4), b])
#bool_idx = (a>3)

#print('\n')
#print(bool_idx)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[2,3,4],[5,6,7],[8,9,10]])
print(np.rank(np.dot(a,b)))
'''
'''n = input()
alist = [int(d) for d in str(n)]

def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

bubbleSort(alist)
alist = [str(k) for k in alist]
print(''.join(alist[::-1]))'''

'''def filter_list(l):
 d= []
 for i in range(len(l)):
  if str(l[i]).isalpha() == True:
      if '\'' in list(l[i]):

       continue
  else:
      d.append(l[i])
 print(d)
filter_list([1,'a','b','0',15])
'''
'''import math
i = 0
for i in range(int(input())):

  c = int(input())
  a = 0
  b = 0
  for b in range(c):
   for a in range(1, b):
     k = 0
     c = math.sqrt(a * a + b * b)
     if c % 1 == 0:
         if 0.5*a*b % 6 != 0 and 0.5*a*b % 28 != 0:
             k += 1
             print(k)
             i += 1
     else:
        i += 1'''

'''import random

i = 0
a = []
for i in range(8):
    a.append(int(input()))

b = a
print(b)
o = []
for i in range(100):
    for k in range(6):
        o.append(random.choice(a))
        b = list(set(a)^set(b))
    if sum(o) == 100:
        for k in range(len(o)):
            print(o[k])
        i += 1
    else:
        i += 1'''

'''
'''

#                            MATRIX WALI PROBLEM


'''import numpy as np
import re
x = 0
a, b = map(int, input().split())
i = 0
char = np.chararray((a,b), unicode = True, itemsize= 10)


while i < a  :
    char[i] = [x for x in input()]
    i += 1

lst = []
i = 0

bhar = char.transpose()
while i< b:
    lst.append(x for x in bhar[i])
    i += 1
flst = []
for x in lst:
    flst.append(''.join(list(x)))
k = ' '.join(flst)

#k = re.sub(r'(?<=[a-z])[@#$%^&()!]+(?=[a-z])', r'', k)
p = k.replace(' ', '')

def repl(m):
    return m.group(1) + ' ' + m.group(2)
y = re.sub(r'([A-Za-z0-9])[^A-Za-z0-9]+([A-Za-z0-9])', repl, p)

print(y)
'''

#                                    CODE CHEF L PROBLEM , DOESNT WORK

'''import numpy as np

ch = np.chararray((3,3), unicode = True, itemsize = 8)

for i in range(3):
 ch[i] = [x for x in input()]

if ch[0,0] == 'l' and ch[1,0] == 'l' and ch[1,1]== 'l' :
    print('yes')
elif ch[1,0] == 'l' and ch[1,1] == 'l' and ch[2,1] == 'l':
    print('yes')
elif ch[0,1] == 'l' and ch[0, 2] == 'l' and ch[1,2] == 'l':
    print('yes')
elif ch[1, 1] == 'l' and ch[1,2] == 'l' and ch[2,2] == 'l':
    print('yes')
else:
    print('no')'''
'''
from bs4 import BeautifulSoup
import requests
import re
import urllib

link = "http://noob31337.blogspot.com/2017/06/paid-courses_30.html?m=1"
r = requests.get(link)

data = str(r.text)
'''
#s = BeautifulSoup(data, 'html.parser')
#lst = []
#for lnk in re.findall(r'^h', data):
 #   lst.append(lnk)'''

#reqlinks = re.findall(r'(https?://\S+)', data)
#print(reqlinks)


################

'''
#                            COMPLETE THIS
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # another classifier , not regressor
from sklearn import linear_model
import numpy as np
from sklearn import metrics

iris = load_iris()

type(iris)

X = iris.data
#iris_X = iris.data[:, np.newaxis, 2]

#iris_X_train = iris_X[:-20]
#iris_X_test = iris_X[-20:]

y = iris.target

#iris_y_train = iris.target[:-20]
#iris_y_test = iris.target[-20:]


#regr = linear_model.LinearRegression()

#regr.fit(iris_X_train, iris_y_train)

#iris_y_pred = regr.predict(iris_X_test)

#print(r2_score(iris_y_test, iris_y_pred))
'''


######## model training - 04
'''print(X.shape)
print(y.shape)

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X, y)

print(knn.predict([[99,99,1,1]]))

logreg = LogisticRegression()

logreg.fit(X, y)

print(logreg.predict([[99,99,1,1]]))'''



######### comparing models or model evaluation

'''
# Using logistic regression
logreg = LogisticRegression()

logreg.fit(X, y)

y_pred = logreg.predict(X)

print(metrics.accuracy_score(y,y_pred))

# Using KNN with n = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

# knn with n = 1
knn1 = KNeighborsClassifier(n_neighbors=1)

knn1.fit(X, y)
y_pred = knn1.predict(X)
print(metrics.accuracy_score(y, y_pred))
'''


######### evaluation procedure 2 - train/test split
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4) #, random_state = 4)

from matplotlib import pyplot as plt

k_range = list(range(1, 26))
scores = []
for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)

y = scores

plt.plot(k_range, scores)
plt.show()'''


############ pandas seaborn scikit-learn, linear regression. 06

'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('Advertising.csv', index_col=0)
#print(data.head(20))



#a = sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', kind='reg')
#plt.show(a)


X = data[['TV', 'radio', 'newspaper']] # outer bracket tells pandas to select a subset of dataframe columns, and the inner brackets form a list

#print(X.head())

y = data['sales'] # used Sales directly; this can only be done if there are no spaces in the name of the column/ Alternatively use y = data['Sales'] , notice that only one set of brackets are used

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 1)

linreg = LinearRegression()

linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print(*zip(X, linreg.coef_)) # coeff of each item required for linear regression

print(metrics.mean_squared_error(y_train, y_pred))
'''

####################################### OPEN CV


import cv2
import numpy as np
#from matplotlib import pyplot as plt

'''img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)# 0 ) can also use numbers. -1 is colour, 0 is grayscale and 1 is unchanged
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([200,300,400],[100,200,300], 'c', linewidth= 5)
plt.xticks([]), plt.yticks([]) # ?? purpose?
plt.show()

cv2.imwrite('gray.png', img)'''

################## Video cap


cap = cv2.VideoCapture(0)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #out.write(frame)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

################## drawing shapes with cv

'''img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(255,255,255),50)
cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
cv2.circle(img,(447,63), 33, (0,255,0), -1)
pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(10,500), font, 2, (200,255,155), 3, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#################### Image operations

'''img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
px = img[55,55]
#cv2.imshow('image', img[100:150, 100:150])

#img[100:150,100:150] = [255,255,255]

watch_face = img[37:111,107:194]
img[0:74, 0:87] = watch_face
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


############ image operations part 2


### Complicated - look up the parameters for each function.

'''img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')

#sum = img1 + img2
#sum = cv2.add(img1,img2)
#weight = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

rows, cols, channels = img3.shape
roi = img1[0:rows, 0:cols]

img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img3gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask) # bitwise is a low level logical operation you can perform, C level stuff

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img3_fg = cv2.bitwise_and(img3,img3, mask=mask)

dst = cv2.add(img1_bg, img3_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res',img1)
#cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

######### THRESHOLDING IN IMAGES

'''img = cv2.imread('1.jpg')

retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

gauss = cv2.adaptiveThreshold(grayscaled, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#cv2.imshow('k', threshold2)
#cv2.imshow('1k', img)
#cv2.imshow('kn', threshold)
cv2.imshow('knf', gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()'''



