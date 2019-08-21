import cv2
import numpy as np



def skeltonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel


# Capturing video from webcam:
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Fixing the mirroring of the current frame
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    skel = skeltonize(gray)


    # Display the resulting frame
    cv2.imshow('frame',skel)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()