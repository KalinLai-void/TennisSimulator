import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from math import atan2, degrees
import math
import socket
import threading
import matplotlib.pyplot as plt
import csv

Ypoint = deque(maxlen=128)
data = []
pointX = []
pointY = []
last_tennis_pos = None
serverIP = "127.0.0.1"
serverPort = 5055
# 設置網球場尺寸
length = 23.6
width = 10.94
text_x = -3
text_y = 4
start_processing = False
start_processing2 = False


def hsv_filter(img):
    # 将BGR颜色空间转成HSV空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义范围 网球颜色范围
    lower_color = (20,120,120)
    upper_color = (60,255,255)

    # 查找颜色
    mask_img = cv2.inRange(hsv_img, lower_color, upper_color)
    return mask_img

def hsv_filter2(img):
    # 将BGR颜色空间转成HSV空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义范围 网球颜色范围
    lower_color = (20,120,130)
    upper_color = (60,255,255)

    # 查找颜色
    mask_img = cv2.inRange(hsv_img , lower_color, upper_color)
    return mask_img

def dilate1(img):
    return cv2.dilate(img, None, iterations=2)


def erode1(img):
    return cv2.erode(img,None, iterations=2)

def dilate2(img):
    return cv2.dilate(img, np.ones((9, 9), np.uint8), iterations=2)


def erode2(img):
    return cv2.erode(img,None, iterations=2)

def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('\n像素坐標 x = %d, y = %d' % (x, y))

def receive_data():
    while True:
        data, address = sock2.recvfrom(1024)
        print(data)
        message = data.decode('utf-8')
        values = message.split(',')
        pointX.append(values[0])
        pointY.append(values[1])

def closest_to_17(a, b):
    if abs(a - 16) < abs(b - 16):
        return a
    else:
        return b

knn = cv2.createBackgroundSubtractorKNN()
knn2 = cv2.createBackgroundSubtractorKNN()
cap1=cv2.VideoCapture("D:/tennis/side2.mp4")
cap2=cv2.VideoCapture("D:/tennis/air2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('line.avi', fourcc, 20.0, (766,  826))

scales = 103.2727272727273 #pixel/公尺 紅土
side_scales = 0.0072916666666667 # 高度比
Zscales = 0.0104575163398693 #紅土
Xscales = 0.0093696763202726 #紅土
air_scales = 0.0261096605744125 # 角度差 / 像素


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5054)

sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddress = (serverIP, serverPort)
sock2.bind(serverAddress)
def capture_frames():
    global start_processing
    frame_id = 0
    pts_0 = deque(maxlen=3)
    pts_1 = deque(maxlen=3)
    while True:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        frame_id += 1
        rows,cols = frame2.shape[:2]
        # pts_o = np.float32([[414,479],[1524,479],[207,970],[1731,979]])
        # pts_d = np.float32([[0,0],[766,0],[0,826],[766,826]])
        pts_o = np.float32([[549,414],[1266,398],[306,907],[1515,873]])
        pts_d = np.float32([[0,0],[766,0],[0,826],[766,826]])
        M = cv2.getPerspectiveTransform(pts_o, pts_d)
        dst = cv2.warpPerspective(frame2, M, (766, 826))

        fgmask1 = knn.apply(frame1)
        fgmask1 = erode1(fgmask1)
        hsv_mask1 = hsv_filter(frame1)
        mask1 = cv2.bitwise_and(fgmask1, hsv_mask1)
        mask1 = dilate1(mask1)
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt1 in contours1:
            area1 = cv2.contourArea(cnt1)
            # if area1 > 10:
            x1, y1, w1, h1 = cv2.boundingRect(cnt1)
            tennis_x1, tennis_y1 = x1 + w1/2 ,y1 + h1/2
            # if len(contours1) > 1:
            frame1 = cv2.circle(frame1, (int(tennis_x1),int(tennis_y1)), 10, (0,0,225), -1)
            pts_0.append([tennis_x1,tennis_y1,frame_id])
            Ypoint.append(tennis_y1)
                # if len(pts_0) > 2 and (pts_0[1][2] == pts_0[2][2]):
                #     min_x = max(pts_0, key=lambda x: x[0])
                #     pts_0.remove(min_x)
                #     if len(pts_0) > 1 and pts_0[0][2] == pts_0[1][2]:
                #         min_x = max(pts_0, key=lambda x: x[0])
                #         pts_0.remove(min_x)


        # fgmask2 = cv2.GaussianBlur(dst, (11, 11), 0)
        fgmask2 = knn2.apply(dst)
        fgmask2  = cv2.medianBlur(fgmask2 , 5)
        hsv2 = hsv_filter2(dst)
        mask2 = cv2.bitwise_and(fgmask2, hsv2)
        ret, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
        mask2 = dilate2(mask2)
        # fgmask2 = cv2.Canny(fgmask2, 20, 160)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i,cnt2 in enumerate(contours2):
            area2 = cv2.contourArea(cnt2)
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            tennis_x2, tennis_y2 = x2 + w2/2 ,y2 + h2/2
            # if len(contours2) > 1:
            dst = cv2.circle(dst, (int(tennis_x2),int(tennis_y2)), 10, (0,0,225), -1)
            pts_1.append([tennis_x2,tennis_y2,frame_id])
            # if len(pts_1) > 2 and  pts_1[1][2] == pts_1[2][2]:
            #     min_y = max(pts_1, key=lambda x: x[1])
            #     pts_1.remove(min_y)
            #     if len(pts_1) > 1 and pts_1[0][2] == pts_1[1][2]:
            #         min_y = max(pts_1, key=lambda x: x[1])
            #         pts_1.remove(min_y)

        if len(pts_0) > 1 and len(pts_1) > 1 and pts_0[0][0] > pts_0[1][0] and pts_1[0][1] > pts_1[1][1]:
            side_start_point = pts_0[1]
            side_end_point = pts_0[0]
            air_start_point = pts_1[0]
            air_end_point = pts_1[1]
            side_y = side_end_point[1] - side_start_point[1]
            side_x = side_end_point[0] - side_start_point[0]
            air_y = air_end_point[1] - air_start_point[1]
            air_x = air_end_point[0] - air_start_point[0]
            ymin = max(Ypoint)
            startY = ((ymin - side_end_point[1]) ) + 0.05
            startY = startY * side_scales
            # if startY < 1.2:
            #     startY = 1.3
            startX = ((pts_1[0][1]) * (-Xscales)) -6.3
            if pts_1[1][0] >= 385 :
                startZ = pts_1[1][0] * (-Zscales) + 5.47
            elif pts_1[1][0] < 385 :
                startZ = pts_1[1][0] * Zscales -5.47
            air_distance = math.dist(pts_1[1], pts_1[0])
            side_angle = 90 - degrees(atan2(side_x, side_y))
            air_angel = degrees(atan2(air_x, air_y))
            if air_angel < 0:
                air_angel = (180 +  air_angel) + ((pts_1[1][0] - 378 ) * air_scales)
            elif air_angel > 0:
                air_angel = -(180 - air_angel)
                # + ((pts_1[1][0] - 378 ) * air_scales)
            time_diff = cap1.get(cv2.CAP_PROP_FPS)
            velocity = (air_distance/scales)/(1/time_diff)
            # if air_id > 0:
            #     velocity = z_dis/(air_id/time_diff) 
            
            if len(data) < 1:
                # cv2.line(frame1, (int(pts_0[0][0]),int(pts_0[0][1])), (int(pts_0[1][0]),int(pts_0[1][1])), (0, 0, 255), 5)
                # cv2.line(dst, (int(pts_1[0][0]),int(pts_1[0][1])), (int(pts_1[1][0]),int(pts_1[1][1])), (0, 0, 255), 5)
                print('pts_0',pts_0)
                print('pts_1',pts_1)
                print('side_角',side_angle)
                print('air_角',air_angel)
                print(f"球的速度為 {velocity:.2f} 公尺/秒")
                print(startX,startY,startZ)
                data.append([side_angle,air_angel,velocity,startX,startY,startZ])
                # data = (side_angle,air_angel,velocity,startX,startY,startZ)
                sock.sendto(str.encode(str(data)), serverAddressPort)
        if len(data) > 0 and  tennis_x1 < 200:
            data.clear()
            pts_0.clear()
            pts_1.clear()
            Ypoint.clear()
            start_processing == False
            start_processing2 = False
            print('data clear')
        # out.write(hsv_mask1)
        cv2.namedWindow('frame1',0)
        cv2.resizeWindow('frame1', 300, 200)
        cv2.imshow('frame1',frame1)
        # cv2.namedWindow('frame2', 0)
        # cv2.resizeWindow('frame2', 640, 480)
        # cv2.imshow('frame2',dst)
        # cv2.setMouseCallback('frame1', onmouse_pick_points)
        # cv2.setMouseCallback('frame2', onmouse_pick_points)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0) 
        elif cv2.waitKey(10) & 0xff == ord('q'):
            break
receive_thread = threading.Thread(target=receive_data)
capture_thread = threading.Thread(target=capture_frames)
receive_thread.start()
capture_thread.start()
capture_thread.join()
sock2.close()

# imgStack = cvzone.stackImages([run_0(cap[0]), run_1(cap[1])], 2, 0.4)
# cv2.imshow("Image", imgStack)
# cv2.setMouseCallback('frame' + str(i), onmouse_pick_points)
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# 設置圖形大小
plt.figure(figsize=(15, 20))

# 畫出場地線
plt.plot([0, length, length, 0, 0], [0, 0, width, width, 0], color='green')

# 畫出中央線
plt.plot([length/2, length/2], [0, width], color='black')

#單打線
plt.plot([length,0],[1.37,1.37],color='green')
plt.plot([length,0],[width-1.37,width-1.37],color='green')

#發球線
plt.plot([5.5,5.5],[0,width],color='green')
plt.plot([length-5.5,length-5.5],[0,width],color='green')

#中線
plt.plot([length-5.5,5.5],[width/2,width/2],color='green')
with open ('output.csv','w',newline="") as csvfile:
    writer  = csv.writer(csvfile)
    for i,j in zip(pointX,pointY):
        plt.scatter(length/2 + float(i),width/2 - float(j), color='Brown')
        writer.writerow([i,j])
    
# 設置標籤和標題
plt.xlabel('Length (m)')
plt.ylabel('Width (m)')
plt.title('Tennis Court')
plt.gca().invert_xaxis()
# 顯示圖形
plt.show()

