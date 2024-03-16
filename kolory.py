import cv2
import math
import numpy as np



def image_load(image_index):
    if j < 10:
        image = cv2.imread('0'+str(j)+'.jpg')
        return image
    elif j >= 10:
        image = cv2.imread(str(j)+'.jpg')
        return image

def HSV_RED(HSV):

    kernel = np.ones((3, 3), np.uint8)
    HSV_lower_1 = np.array([0,130,40])
    HSV_higher_1 = np.array([5,255,255])
    HSV_mask_1 = cv2.inRange(HSV, HSV_lower_1, HSV_higher_1)

    kernel = np.ones((3, 3), np.uint8)
    HSV_lower_2 = np.array([175,120,10])
    HSV_higher_2 = np.array([185,255,255])
    HSV_mask_2 = cv2.inRange(HSV, HSV_lower_2, HSV_higher_2)
  
    Hmask = cv2.erode(HSV_mask_1+HSV_mask_2, kernel, iterations=3)
    Hmask = cv2.dilate(Hmask, kernel,iterations=2)
    return Hmask

def HSV_GREEN(HSV):
    kernel = np.ones((3, 3), np.uint8)
    HSV_lower_1 = np.array([28,185,20])
    HSV_higher_1 = np.array([65,255,255])
    HSV_mask_1 = cv2.inRange(HSV, HSV_lower_1, HSV_higher_1)

    kernel = np.ones((3, 3), np.uint8)
    HSV_lower_2 = np.array([40, 45, 90])
    HSV_higher_2 = np.array([53, 190, 255])
    HSV_mask_2 = cv2.inRange(HSV, HSV_lower_2, HSV_higher_2)
 
    Hmask = cv2.erode(HSV_mask_1+HSV_mask_2, kernel,iterations=3)
    Hmask = cv2.dilate(Hmask, kernel)
    # cv2.imshow('img',Hmask)
    # cv2.waitKey(0)  
    # Hmask = cv2.morphologyEx(Hmask, cv2.MORPH_OPEN, kernel)
    return Hmask

def HSV_YELLOW(HSV):
    kernel = np.ones((3, 3), np.uint8)
    HSV_lower = np.array([18,180,62])
    HSV_higher = np.array([28,255,251]) 

    Hmask = cv2.inRange(HSV, HSV_lower, HSV_higher)
    
    Hmask = cv2.erode(Hmask, kernel, iterations=3)
    Hmask = cv2.dilate(Hmask, kernel,iterations=2)
 

    return Hmask


def HSV_PURPLE(HSV):

    kernel = np.ones((3, 3), np.uint8)
    HSV_lower_1 = np.array([150, 30, 20]) 
    HSV_higher_1 = np.array([173, 255, 245]) 
    HSV_mask_1 = cv2.inRange(HSV, HSV_lower_1, HSV_higher_1)    

    kernel = np.ones((3, 3), np.uint8)    
    HSV_lower_2 = np.array([0, 0, 0])
    HSV_higher_2 = np.array([40, 100, 70])
    HSV_mask_2 = cv2.inRange(HSV, HSV_lower_2, HSV_higher_2)
 
 
    Hmask = cv2.erode(HSV_mask_1+HSV_mask_2, kernel, iterations=3)
    Hmask = cv2.dilate(Hmask, kernel,iterations=3)
    return Hmask



def finding_contours(img, type):
        # blur = cv2.GaussianBlur(img,(5,5),0)  
        # edges = auto_canny(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        counter = 0      
        for cnt in contours:         
            area = cv2.contourArea(cnt)
            # print(area)
            if area > 100 and type == 1:
                cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
                counter = counter + 1
            elif area > 66  and type == 2:
                cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
                counter = counter + 1
            elif area > 35 and type == 3 :
                cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
                counter = counter + 1
            elif area > 100 and type == 4:
                cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
                counter = counter + 1
        return counter



i = 0
for j in range(40):
    # Let's load a simple image with 3 black squares
            IMAGE = image_load(j)
            #IMAGE = cv2.imread('12.jpg')
            scale = IMAGE.shape[0] / 1000
            w = int(IMAGE.shape[1] / scale)
            h = int(IMAGE.shape[0] / scale)
            dimension = (w, h)
            IMAGE = cv2.resize(IMAGE, dimension, interpolation=cv2.INTER_LANCZOS4)
            HSV = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2HSV)
            
            red_hsv = HSV_RED(HSV)
            image_red = cv2.bitwise_and(IMAGE,IMAGE, mask=red_hsv)
            r = finding_contours(image_red, 4)

            green_hsv = HSV_GREEN(HSV)
            # cv2.imshow('img',green_hsv)
            # cv2.waitKey(0)            
            image_green = cv2.bitwise_and(IMAGE,IMAGE, mask=green_hsv)
            g = finding_contours(image_green, 1)
            # cv2.imshow('img',image_green)
            # cv2.waitKey(0)  

            yellow_hsv = HSV_YELLOW(HSV)
            image_yellow = cv2.bitwise_and(IMAGE,IMAGE, mask=yellow_hsv)
            y = finding_contours(image_yellow, 3)

            purple_hsv = HSV_PURPLE(HSV)   
            image_purple = cv2.bitwise_and(IMAGE,IMAGE, mask=purple_hsv)
            p = finding_contours(image_purple, 2)

            print(" g :"+str(g)+" p :"+str(p)+" y :"+str(y)+" r :"+ str(r) + " index : " + str(j))
            # cv2.imshow('img',IMAGE)
            # cv2.imshow('GREEN',image_green)
            # cv2.waitKey(0)
            # cv2.imshow('RED',image_red)
            # cv2.imshow('Yellow',image_yellow)
            # cv2.imshow('Purple',image_purple)
            # cv2.waitKey(0)
            
