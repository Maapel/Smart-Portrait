import cv2
import dlib
import faceBlendCommon as fbc
import numpy as np
import math
import random
from cvzone.FaceMeshModule import FaceMeshDetector

src = cv2.VideoCapture(0)
# detector2 = FaceMeshDetector(maxFaces=3)

i=0
size = [4,4]

# Skin mouth hair



while cv2.waitKey(1)!=27:
    i+=1
    ret, img = src.read()
    img0= cv2.GaussianBlur(img,(7,7),0)
    img= img0
    # img0 = img
    # ret = Tru
    # e
    # img0 = cv2.imread("img1.jpg")
    # img = img0
    if not(ret):
        break

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    img_blur = cv2.medianBlur(img,9)
    final = cv2.Canny(img_blur,60,100)
    # final = img_blur
    # final2 = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,12 )

    img2=img

    # for face in faces:
    #     cv2.rectangle(img2,(face[0],face[1]),(face[2]+face[0],face[3]+face[1]),(0,0,4),2)

    # img2, faces = detector2.findFaceMesh(img)
    cv2.imshow("Face detected", img)


    # cv2.imshow("threshol", img)
src.release()
cv2.destroyAllWindows()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Read array of corresponding points
# from the faces
points1 = fbc.getLandmarks(detector, predictor, img)

hullIndex = cv2.convexHull(np.array(points1).astype(np.int32), returnPoints=False)
# Create convex hull lists
hull1 = []
for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])
hull_outer = []
for i in range(0, len(hull1)):
    hull_outer.append((hull1[i][0], hull1[i][1]))
mask = np.zeros(img.shape, dtype=img.dtype)
cv2.fillConvexPoly(mask, np.int32(hull_outer), (255, 255, 255))
img3 = np.bitwise_and(mask , img)

tempImg = img.copy()
tempImg = cv2.GaussianBlur(tempImg,(9,9),0)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(img,mask)
img2_fg = cv2.bitwise_and(tempImg,mask_inv)
img_n = cv2.add(img1_bg,img2_fg)

img = img_n
face=faces[0]
img = img[face[1]-face[3]//4:,face[0]-face[2]//6:7*face[2]//6+face[0]]


a =0
new  = np.zeros(img.shape, dtype=img.dtype)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgn = img


########################################################################################
######################################################################################
base_colors = [[172, 190, 232],[106,106,231],[255,30,30],[0,40,40],[220,220,220]]
colors_rgb = [[39, 31, 230],[200, 39, 36],[100,140,200]]
colors = np.array(colors_rgb)
colors = colors[::,::-1]

colors = colors.tolist()
clrs = []
def avg(c1,c2):
    del_r = c2[2] - c1[2]
    del_g = c2[1] - c1[1]
    del_b = c2[0] - c1[0]
    return abs(del_b+del_g + del_r)/3

def diff(c1,c2):
    r_m = (c1[2]+c2[2])//2
    del_r = c2[2]-c1[2]
    del_g = c2[1]-c1[1]
    del_b = c2[0]-c1[0]
    return pow((2+r_m/256)*pow((del_r),2)+4*pow(del_g,2)+(2+(255-r_m)/256)*pow(del_b,2),0.5)
    # return pow(pow((del_r),2)+pow(del_g,2)+pow(del_b,2),0.5)

clrs = colors
# for b in base_colors:
#     min = [diff(colors[0], base_colors[0]), colors[0]]
#     for c in colors:
#         d = diff(c, b)
#         if d <= min[0]:
#             min = [d, c]
#     colors.remove(min[1])
#     for i in range(3):
#
#         if (min[1][i]+int(avg(min[1],b)) in range(0,255)):
#             min[1][i] = min[1][i] + int(avg(min[1], b))
#
#     clrs.append(min[1])
# clrs = colors[0:3]
def get_color(c):
    # min = [diff(clrs[0],c),0]
    # for clr in range(len(clrs)):
    #     d = diff(c,clrs[clr])
    #     if d<min[0]:
    #         min = [d,clr]

    delta = 0
    color = clrs[random.randint(0,len(colors)-1)]

    for a in range(3):
        delta += c[a]-color[a]
    delta=delta//3
    # print("original to", c, clrs[min[1]])
    res=[]
    for a in color:
        if (a+delta<0)or(a+delta>255):
            res.append(a)
            print("yes",a)
        else:
            res.append(a+delta)
    print(res)
    return res
print("User colors ",clrs)
patches= []
patches_gray= []

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(img,80,90)
edge = cv2.cvtColor(edge , cv2.COLOR_GRAY2BGR)

for i in range(0,np.size(img,0)):
    for j in range(0,np.size(img,1)):
        color = []
        step = 33
        # for c in range(3):
        #     clr = step*(img[i][j][c]//step)
        #     img[i][j][c] = clr
        #     color.append(clr)
        a = step*(gray[i][j]//step)
        color = [a,a,a]
        if not(color in patches_gray):
            patches_gray.append(color)
            patches.append(img[i][j])
        else:
            img[i][j] = patches[patches_gray.index(color)]



cols = []
for  p in patches:
    cols.append(get_color(p))
mask = [(img == c).all(axis=-1)[..., None] for c in patches]
img = np.select(mask,cols , img)
print(img)
img = np.array(img,dtype=img0.dtype)
# img = cv2.add(img ,edge//4)
for i in range(0,np.size(edge,0)):
    for j in range(0,np.size(edge,1)):
        if edge[i][j][0]!=0:
            img [i][j]=img[i][j]//10
# img  = img -edge//10
# img = cv2.add(img,img2_fg)
# img = cv2.subtract(img ,edge//9)

cv2.imshow("edges",edge)
cv2.imshow("final", img)
cv2.imshow("blurred",img_n)
cv2.waitKey()