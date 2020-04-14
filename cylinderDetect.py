#!/usr/bin/env python

import os
import argparse
import rospy
import cv2
import numpy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from sklearn.linear_model import LogisticRegression

class LogRegression:
    def __init__(self, img, mask):
        self.img = img #Original image
        self.mask = mask #Mask of image
        self.bridge = CvBridge()
        self.current = []

        grayMask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        (thresh,binMask) = cv2.threshold(grayMask, 127, 255, cv2.THRESH_BINARY)

        self.logr = LogisticRegression(class_weight = 'balanced', solver='lbfgs')
        data = img.reshape((-1,3)).astype(float)
        label = (binMask.ravel()>0).astype(int)
        self.logr.fit(data,label)

        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.display_compressed)
    
    def display_compressed(self, data):
        try:
            self.current = self.bridge.compressed_imgmsg_to_cv2(data,'bgr8')
        except CvBridgeError as e:
            print(e)
        
        probImg = self.regression()
        cv2.imshow("Original Image", self.current.copy())
        cv2.imshow("Probability", numpy.array(probImg.copy(),dtype = numpy.uint8))
        cv2.waitKey(1)
    
    def regression(self):
        
        if len(self.current) > 0:
            vData = numpy.reshape(self.current,(-1,3)).astype(float)
            pixProb = self.logr.predict_proba(vData)

            probImg = []
            for x in pixProb:
                if x[1] > 0.9:
                    probImg.append(255)
                else:
                    probImg.append(0)
            probImg = numpy.reshape(probImg,(480,640))

            return probImg
        else:
            return 0

    
            
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image and Mask')
    parser.add_argument('imgName', type=str, metavar='PATH', help='Full path of image name')
    parser.add_argument('maskName', type=str, metavar='PATH', help='Full path of mask name')

    args = parser.parse_args()

    if os.path.isfile(args.imgName):
        print("Reading in image:", args.imgName)
        img = cv2.imread(args.imgName)
    else:
        print("Error: unable to find:", args.imgName)

    if os.path.isfile(args.maskName):
        print("Reading in image:", args.maskName)
        mask = cv2.imread(args.maskName)
    else:
        print("Error: unable to find:", args.maskName)
    
    rospy.init_node('regression', anonymous=True)

    lr = LogRegression(img, mask)
    lr.regression()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()