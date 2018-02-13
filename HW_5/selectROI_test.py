import numpy as np
import cv2

if __name__ == '__main__' :

    # Read image
    im = cv2.imread("image.png")

    # Select ROI
    r = cv2.selectROI(im)

    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)

# cap = cv2.VideoCapture(1)
#
# ## this section was used for viewing image in greyscale
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Select ROI
#     image = cv2.imread(frame)
#     r = cv2.selectROI(image)
#
#     # Crop image
#     imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#
#     # Display cropped image
#     cv2.imshow("Image", imCrop)
#     cv2.waitKey(0)
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     src = frame
#     ksize = (15,15)
#     sigmaX = 0
#     dst = 'blurred'
#     sigmaY = 1
#     borderType = 'BORDER_TRANSPARENT'
#     blur = cv2.GaussianBlur(src,ksize,sigmaX)#[,dst[,sigmaY[,borderType]]])
#     # color = cv2.cvtColor(frame, cv2.COLOR)
#     # Display the resulting frame
#     cv2.imshow('GausBlur',frame)
#     # cv2.imshow('frame')
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
