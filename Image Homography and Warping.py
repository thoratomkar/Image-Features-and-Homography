# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:11:55 2018

@author: omkar
"""

import cv2
import numpy as np
import random

def key_point_detection(img1,s):
    
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1= sift.detectAndCompute(gray1,None)
    gray1=cv2.drawKeypoints(gray1,kp1,gray1)
    cv2.imwrite('task1'+'_sift'+s+'.jpg',gray1)
    
    return gray1,kp1,des1

def key_point_matching(img1,img2,kp1,kp2,des1,des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    good2=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good2.append(m)
        
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img1,flags=2)
    cv2.imwrite('task1_matches_knn.jpg',img3)
    
    MIN_MATCH_COUNT=10    
    if len(good2)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good2 ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good2 ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)        
        matchesMask = mask.ravel().tolist()       
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None    
    
    #random selection of 10 pairs
    x = random.randint(0,len(matchesMask))
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask[x:x+10], # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good2[x:x+10],None,**draw_params)
    cv2.imwrite('task1_matches.jpg',img3)
    
    return img3,H


def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img2
	return output_img


img_1 = cv2.imread('mountain1.jpg')
img1,kp1,des1 = key_point_detection(img_1,'1')
img_2 = cv2.imread('mountain2.jpg')
img2,kp2,des2 = key_point_detection(img_2,'2')
img3,H = key_point_matching(img1,img2,kp1,kp2,des1,des2)
print(H)
result = warpImages(img_1, img_2, H)
cv2.imwrite('task1_pano.jpg',result)
