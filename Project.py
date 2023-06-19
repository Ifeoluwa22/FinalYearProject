import cv2
import mediapipe as mp
from mediapipeai.PosEstimationModule import poseDetector
import os
import cvzone

cap = cv2.VideoCapture(0)
detector= poseDetector()

listShirts = os.listdir("Resources")
print(listShirts)
fixedRatio = 262/190 #widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 440/380
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    img = cv2.flip(img,1)
    lmList = detector.getPosition(img)
    if lmList:
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        detector.showFps(img)
        imgShirt = cv2.imread(os.path.join("Resources", listShirts[2]), cv2.IMREAD_UNCHANGED)
        
        widthOfShirt = int((lm11[0]-lm12[0])*fixedRatio)
        print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt,int(widthOfShirt*shirtRatioHeightWidth)))    
        currentScale = (lm11[0]-lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)
        # print(lmList)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0]+offset[0], lm12[1]+offset[1]))
        except:
            pass
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)


