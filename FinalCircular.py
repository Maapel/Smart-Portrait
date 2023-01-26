import cv2
#import dlib
#import faceBlendCommon as fbc
import numpy as np
import math
import random
import os
#from cvzone.FaceMeshModule import FaceMeshDetector
import time
src = cv2.VideoCapture(0)
# detector2 = FaceMeshDetector(maxFaces=3)

i=0
#size = [4,4]

# Skin mouth hair

from cvzone.HandTrackingModule import HandDetector
import cv2

import time
from cvzone.FaceMeshModule import FaceMeshDetector


detector2 = FaceMeshDetector(maxFaces=3)

total_count = 0

gest = cv2.imread("gesture.png")
detector = HandDetector(detectionCon=0.8, maxHands=1)


def capture(p, center, r):
    x, y = p
    a, b = center
    dist = ((x - a) ** 2 + (y - b) ** 2) ** (1 / 2)
    if dist <= r:
        return True
    return False

def avg(c1, c2):
    del_r = c2[2] - c1[2]
    del_g = c2[1] - c1[1]
    del_b = c2[0] - c1[0]
    return abs(del_b + del_g + del_r) / 3


def diff(c1, c2):
    r_m = (c1[2] + c2[2]) // 2
    del_r = c2[2] - c1[2]
    del_g = c2[1] - c1[1]
    del_b = c2[0] - c1[0]
    return pow((2 + r_m / 256) * pow((del_r), 2) + 4 * pow(del_g, 2) + (2 + (255 - r_m) / 256) * pow(del_b, 2), 0.5)
    # return pow(pow((del_r),2)+pow(del_g,2)+pow(del_b,2),0.5)

R=20
radius = 50
margin = 10
x0=radius+margin
y0=radius+margin
x_max=500
x=x0
y=y0
max_colors=1


def select_box(p):
    x, y = p

    for circle in circles:
        a = circle[0]
        b = circle[1]
        dist = ((x - a) ** 2 + (y - b) ** 2) ** (1 / 2)
        if dist <= radius:
            if circle in selected:
                selected.remove(circle)
            else:
                if len(selected) >= max_colors:
                    selected.remove(selected[0])
                selected.append(circle)

while cv2.waitKey(1)!=27:

    prev=0
    prev_pos =(0,0)
    r=0
    circles=[]

    # To be defined
    thresh_t=1.1
    color_grps =[[[216, 103, 33],[212, 89, 102],[242,212,240]],[[255, 223, 211],[254, 200, 216],[224,187,228]],[[39, 31, 230],[200, 39, 36],[100,140,200]],[[245, 220, 200],[245, 240, 170],[200,239,247]],[[0, 255, 255],[0, 191, 255],[191,0,255]],[[236,150,250],[130,210,236],[120,105,245]]]
    bg = (255,255,255)
    for colors_rgb in color_grps:
        colors = np.array(colors_rgb)
        colors = colors[::, ::-1]

        print(colors)
        # if x>x_max:
        #     x=x0
        arr=[x,y,colors]
        y += 2*radius + margin*2
        circles.append(arr)


    selected=[]





    run = True
    capture_bool = False
    capture_r = 30
    capture_counter = 100

    while run:

        select = False
        # Get image frame
        success, img0 = src.read()
        img0 = cv2.flip(img0, 1)
        # Find the hand and its landmarks
        img0 = cv2.GaussianBlur(img0, (7, 7), 0)
        img = img0.copy()
        if capture_bool:
            capture_counter-=1
            angle1+=3.6
            cv2.ellipse(img, (WIDTH//2,HEIGHT//2 ), (50,50), 0, 0,360-angle1,(255,255,255), -1)
            cv2.ellipse(img, (WIDTH//2,HEIGHT//2 ), (55,55), 0, 0,360 - angle1,(0,0,0), 5)

            cv2.imshow("Select color", img)
            cv2.waitKey(1)
            if capture_counter<=0:
                gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

                faceCascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    face = faces[0]
                    run = False
                    height = HEIGHT - (face[1] - face[3] // 4)
                    width =int(height/1.41)
                    print(width,height)
                    face_x = face[0]+face[2]//2
                    img0 = img0[0:height, face_x - width//2: width//2 + face_x]
                    cv2.resize(img0,(1754,2480))
                    print(img0)

                else:
                    capture_counter=100
                    run = True
                    capture_bool=False
                    print("no face detected")

            continue


        hands= detector.findHands(img,draw=False)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        WIDTH, HEIGHT = (np.size(img, 1), np.size(img, 0))
        capture_pos = (WIDTH // 2, margin +capture_r)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            length, info = detector.findDistance(lmList1[4][0:2], lmList1[8][0:2])



            det = False
            fingers1 = detector.fingersUp(hand1)
            if (fingers1 ==[0,1,1,0,0]):
                pos = ((lmList1[8][0] + lmList1[12][0]) // 2, (lmList1[8][1] + lmList1[12][1]) // 2)
                det = True
            elif (fingers1 == [1, 1, 1, 0, 0]):
                pos = ((lmList1[8][0] + lmList1[12][0]) // 2, (lmList1[8][1] + lmList1[12][1]) // 2)
                det = True

            elif (fingers1 == [0, 1, 0, 0, 0]):
                pos = ((lmList1[8][0] ), (lmList1[8][1]))
                det = True
            elif (fingers1 == [1, 1, 0, 0, 0]):
                pos = ((lmList1[8][0]), (lmList1[8][1]))
                det = True
            else:
                alpha = 0.4
                overlay = np.zeros(img.shape, img.dtype)

                gest_pos = [3 * WIDTH // 8, HEIGHT // 2]
                gest = cv2.resize(gest, (WIDTH // 4, WIDTH // 4))
                overlay[gest_pos[1]:gest_pos[1] + gest.shape[0], gest_pos[0]:gest_pos[0] + gest.shape[1], :] = gest
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            if det:
                dist, inf = detector.findDistance(pos, prev_pos)
                prev_pos = pos
                cv2.circle(img, pos, R, (255, 200, 200), -1)
                if (dist<=8):
                    if r == 0:
                        t=time.time()
                        print("t" , t)
                    print(time.time())
                    delta = time.time()-t
                    print("delta", delta)
                    r = 1+int(R * delta / thresh_t)
                    print(r)
                    cv2.circle(img, pos, r, (125, 25, 25), -1)
                    if delta>=thresh_t:
                        select = True
                        select_box(pos)
                        if capture(pos,capture_pos,capture_r)&len(selected)!=0:
                            capture_bool =True
                            angle1 = 0



                        r=0
                        cv2.circle(img, pos, R, (255, 200, 200), -1)




                else:
                    r=0
        else:


            alpha = 0.4
            overlay = np.zeros(img.shape, img.dtype)

            gest_pos = [3*WIDTH//8, HEIGHT // 2]
            gest = cv2.resize(gest, (WIDTH // 4, WIDTH // 4))
            overlay[gest_pos[1]:gest_pos[1] + gest.shape[0], gest_pos[0]:gest_pos[0] + gest.shape[1], :] = gest
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        for e in circles:
            alpha = 0.9
            overlay = img.copy()
            if e in selected:
                cv2.ellipse(overlay,(e[0],e[1]), (radius,radius),0, 0, 360, (10,10,10), 4)

                  # A filled rectangle
            # cv2.ellipse(overlay, (e[0], e[1]), (radius, radius), 0, 0, 360, (255, 255, 255), -1)

            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            x = e[0]
            y= e[1]
            angle = 0
            for clr in e[2]:

                overlay = img.copy()
                color = (int(clr[0]),int(clr[1]),int(clr[2]))

                cv2.ellipse(overlay, (e[0], e[1]), (radius-5, radius-5), 0, angle, angle+120, color, -1)
                # A filled rectangle
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                angle+=120

        overlay = img.copy()

        alpha = 0.9
        cv2.circle(overlay,capture_pos,capture_r-5,(255,255,255),-1)
        cv2.circle(overlay,capture_pos,capture_r,(255,255,255),3)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.imshow("Select color", img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()



    img= img0
    cv2.imwrite("/original/"+str(total_count)+".png",img0)

    new  = np.zeros(img.shape, dtype=img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgn = img


    ########################################################################################
    ######################################################################################
    #base_colors = [[172, 190, 232],[106,106,231],[255,30,30],[0,40,40],[220,220,220]]
    # colors_rgb = [[245, 220, 200],[245, 240, 170],[246,135,145 ]]
    # colors = np.array(colors_rgb)
    colors = selected[0][2]
    print(colors)
    # colors = colors.tolist()
    clrs = []

    clrs = colors
    count = 0
    def get_color(c):

        # min = [diff(clrs[0],c),0]
        # for clr in range(len(clrs)):
        #     d = diff(c,clrs[clr])
        #     if d<min[0]:
        #         min = [d,clr]

        delta = 0
        color = clrs[count%len(clrs)]
        for a in range(3):
            delta += (c[a] - color[a])
        delta=delta//3
        # delta = delta**(1/2)
        print("color",c,delta)
        # print("original to", c, clrs[min[1]])
        res=[]
        for a in color:
            if (a+delta>255):
                res.append(255)
            elif (a + delta <0):
                res.append(0)

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

            a = step*(gray[i][j]//step)
            color = [a,a,a]
            if not(color in patches_gray):
                patches_gray.append(color)
                patches.append(img[i][j])
            else:
                img[i][j] = patches[patches_gray.index(color)]


    gry = img
    cols = []
    for p in patches:
        cols.append(get_color(p))
        count+=1

    mask = [(img == c).all(axis=-1)[..., None] for c in patches]
    img = np.select(mask,cols , img)
    img = np.array(img,dtype=img0.dtype)
    # img = cv2.add(img ,edge//4)
    for i in range(0,np.size(edge,0)):
        for j in range(0,np.size(edge,1)):
            if edge[i][j][0]!=0:
                img [i][j]=(6*img[i][j])//7




    #LOGO
    hh, ww , tt= img.shape

    l = cv2.imread('shaastra_logo1.png',cv2.IMREAD_UNCHANGED)
    h,w,_=l.shape
    logo=cv2.resize(l,(ww//2,(ww//2)*h//w))
    # print(logo)
    h, w ,_= logo.shape
    # print(h,w)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    # load background image as grayscale
    # back = cv2.imread('background.jpeg', cv2.IMREAD_GRAYSCALE)
    #back1=cv2.resize(back,(160,136))


    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round(hh-h)
    xoff = round(ww-w)

    # use numpy indexing to place the resized image in the center of background image
    # result = img.copy()
    # result[yoff:yoff+h, xoff:xoff+w] = logo


    # logo[:,:,:3] = logo[:,:,:3]
    background = img

    overlay = logo

    # separate the alpha channel from the color channels
    alpha_channel = overlay[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
    overlay_colors = overlay[:, :, :3]


    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))



    background_subsection = background[yoff:yoff+h, xoff:xoff+w]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[yoff:yoff+h, xoff:xoff+w] = composite
    cv2.imwrite(str(total_count)+str(time.time())+".png",background)

    cv2.imshow("f", background)
    total_count+=1

cv2.waitKey()
cv2.destroyAllWindows()



