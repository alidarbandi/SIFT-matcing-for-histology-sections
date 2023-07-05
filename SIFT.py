# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:09:29 2023

@author: sem
"""

import numpy as np
import cv2                  ########### this is based on opencv version 4.7.0

from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 5



img1 = cv2.imread('C:/Users/sem/Desktop/temp/target-monkey.jpg', 0)  # Template image
img2 = cv2.imread('C:/Users/sem/Desktop/temp/lowmag-monkey.jpg', 0)  # Main image

#img1 = cv2.resize(img1, None, fx=0.3, fy=0.3)
#img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)


sift = cv2.SIFT_create()  # see Daniel Azemar; https://stackoverflow.com/questions/37039224/attributeerror-module-object-has-no-attribute-xfeatures2d-python-opencv-2

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

print(matches)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w, = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    x, y, w, h = cv2.boundingRect(np.int32(dst))

    # Draw a rectangle around the matched location
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 3)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img2, 'gray'),plt.show()


print(cv2.__version__)































