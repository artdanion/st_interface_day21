import serial
import json
import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

global mX, mY, oldX, oldY, slalingX, scalingY
global Moverange, threshold, center, minimaX, minimaY, maximaX, maximaY

mX = 0
mY = 0
oldX = 0
oldY = 0
Moverange = 12;
threshold = Moverange // 4
center = Moverange // 2

minimaX = 1023
minimaY = 1023
maximaX = 0
maximaY = 0

scalingX = 10
scalingY = 10

def empty(a):
    pass

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def mapping(XYval):
    
    global Moverange, threshold, center, minimaX, minimaY, maximaX, maximaY
    
    valX = int(XYval[0])
    valY = int(XYval[1])
    
    print(valY)
    
    if valX < minimaX:
        minimaX = valX
    if valY < minimaY:
        minimaY = valY
        
    if valX > maximaX:
        maximaX = valX
    if valY > maximaY:
        maximaY = valY
        
        #map(minimaX, maximaX, 0, 1023)
        
    valX = int((1023)*((valX-minimaX)//(maximaX-minimaX)))
    valY = int((1023)*((valY-minimaY)//(maximaY-minimaY)))
    
    if(abs(valX > threshold)):
        valX = (valX - center)
        
    if(abs(valY > threshold)):
        valY = (valY - center)
        
    valY = (valY *(-1))
    
    sendXY = [valX, valY]
    
    dicM = { 'Move':sendXY }
    jSent= json.dumps(dicM)
    print(jSent)
    ser.write(jSent.encode('ascii'))
    return

cam = PiCamera()
cam.resolution = (720, 480)#(512, 304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(720,480)) #(512, 304))

ser = serial.Serial('/dev/ttyUSB0', 19200, timeout=1)
ser.reset_input_buffer()

#cv2.namedWindow("TrackBars", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("TrackBars", 640, 40)
#cv2.createTrackbar("Hue Min", "TrackBars", 48, 179, empty)
#cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
#cv2.createTrackbar("Sat Min", "TrackBars", 136, 255, empty)
#cv2.createTrackbar("Sat Max", "TrackBars", 190, 255, empty)
#cv2.createTrackbar("Val Min", "TrackBars", 134, 255, empty)
#cv2.createTrackbar("Val Max", "TrackBars", 190, 255, empty)


while True:

    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = frame.array
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 0 37 77 158 208 255
        h_min=0
        h_max=37
        s_min=77
        s_max=158
        v_min=208
        v_max=255
        #h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        #h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        #s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        #s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        #v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        #v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        #print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>50:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(img, cnt, -1, (255, 0, 0),2)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
                cv2.line(img,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.line(img,(x,y+h),(x+w,y),(0,255,0),1)
                mX = (x + h//2)
                mY = (y + w//2)
                
                if mX != oldX or mY != oldY:
                    jMove = (oldX-mX), (oldY-mY)
                    mapping(jMove)
                    oldX = mX
                    oldY = mY
                
        
        imgStack = stackImages(0.4, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("Stacked Images", imgStack)
        cv2.imshow("Tracking", img)
        rawCapture.truncate(0)
    
        k = cv2.waitKey(1)
        rawCapture.truncate(0)
        if k%256 == 27: # ESC pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/"+ name +"/img_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            img_counter += 1
            
    if k%256 == 27:
        print("Escape hit, closing...")
        break

cv2.destroyAllWindows()