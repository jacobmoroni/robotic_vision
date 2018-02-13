import numpy as np
import cv2
import sys
colorvid = False #this has to be set to false to record the edge detection. but when it is false it will only record other filters when greyscale is turned on
if len(sys.argv) ==2:
    outfile = sys.argv[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outfile,fourcc, 20.0, (640,480) ,colorvid)
    savefile = True
else:
    savefile = False

cap = cv2.VideoCapture(0)
    #implement sliders
def nothing(x):
    pass
cv2.namedWindow('image')
# create trackbars for color change
switch0 = 'greyscale (on/off)'
cv2.createTrackbar(switch0,'image',0,1,nothing)

switch = 'GausBlur (on/off)'
cv2.createTrackbar(switch, 'image',0,1,nothing)
cv2.createTrackbar('ksize H','image',1,300,nothing)
cv2.createTrackbar('ksize V','image',1,300,nothing)

switch2 = 'EdgeDetect (on/off)'
cv2.createTrackbar(switch2, 'image',0,1,nothing)
cv2.createTrackbar('minval','image',0,300,nothing)
cv2.createTrackbar('maxval','image',0,500,nothing)

switch3 = 'BilateralBlur (on/off)'
cv2.createTrackbar(switch3, 'image',0,1,nothing)
cv2.createTrackbar('diameter','image',0,100,nothing)
cv2.createTrackbar('sigmaColor','image',0,500,nothing)
cv2.createTrackbar('sigmaSpace','image',0,500,nothing)

ret, frame = cap.read()
src = frame
## this section was used for viewing image in greyscale
while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    src = cv2.flip(src,1)
    cv2.imshow('image',src)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    if cv2.getTrackbarPos('ksize H','image') %2 == 1:
        ksizeH = cv2.getTrackbarPos('ksize H','image')
    else:
        ksizeH = cv2.getTrackbarPos('ksize H','image')+1
    if cv2.getTrackbarPos('ksize H','image') <=0:
        ksizeH = 1;
    if cv2.getTrackbarPos('ksize V','image') %2 == 1:
        ksizeV = cv2.getTrackbarPos('ksize V','image')
    else:
        ksizeV = cv2.getTrackbarPos('ksize V','image')+1
    if cv2.getTrackbarPos('ksize V','image') <=0:
        ksizeV = 1;
    # ksizeV = cv2.getTrackbarPos('ksize V','image')
    minval = cv2.getTrackbarPos('minval','image')
    maxval = cv2.getTrackbarPos('maxval','image')
    s0 = cv2.getTrackbarPos(switch0,'image')
    s = cv2.getTrackbarPos(switch,'image')
    s2 = cv2.getTrackbarPos(switch2,'image')
    s3 = cv2.getTrackbarPos(switch3,'image')
    ksize = (ksizeH,ksizeV)

    dia = cv2.getTrackbarPos('diameter','image')
    sigCol = cv2.getTrackbarPos('sigmaColor','image')
    sigSpace = cv2.getTrackbarPos('sigmaSpace','image')

    if s0 ==1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bilsrc = frame
    if s == 0 and s2 ==0 and s3 == 0:
        src = frame
        edgesrc = frame
        bilsrc = frame
    elif s == 0 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 0 and s3 == 0:
        src = blur
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = blur
        bilsrc = frame
    elif s == 0 and s2 == 0 and s3 == 1:
        src = bilblur
        edgesrc = frame
        bilsrc = frame

    elif s == 0 and s2 == 1 and s3 == 1:
        src = edges
        edgesrc = bilblur
        bilsrc = frame
    else:
        src = edges
        edgesrc = blur
        bilsrc = frame

    # if s2 == 1 or s0 ==1:
    #     colorvid = False
    # else:
    #     colorvid = True
    #setup and execution of GaussianBlur
    blursrc = frame
    sigmaX = 0
    # sigmaY = 1
    # borderType = 'BORDER_TRANSPARENT'

    #setup GaussianBlur, Canny edge detection and bilateral blur filters
    blur = cv2.GaussianBlur(blursrc,ksize,sigmaX)#[,dst[,sigmaY[,borderType]]])
    bilblur = cv2.bilateralFilter(bilsrc,dia,sigCol,sigSpace)
    edges = cv2.Canny(edgesrc, minval, maxval)
    if savefile == True:
        out.write(src)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
