import numpy as np
import cv2

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
if __name__ == '__main__' :

    # Read image
    im = cv2.imread("image.png")

    # Select ROI
    r = cv2.selectROI(im)
    mask = np.zeros_like(im)
    mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]),:] = 255
    print mask
    p = cv2.goodFeaturesToTrack(im, mask = mask, **feature_params)
    print (p)
    # Crop image
    # imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Display cropped image
    # cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)

# cap = cv2.VideoCapture('mv2_001.avi')
# # Define an initial bounding box
# bbox = (287, 23, 86, 320)
# ret, frame = cap.read()
# # Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)
# ## this section was used for viewing image in greyscale
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Select ROI
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
