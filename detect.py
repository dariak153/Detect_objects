import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm


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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        counter = 0      
        for cnt in contours:         
            area = cv2.contourArea(cnt)
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








def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.

    s = img.shape[0] / 1000
    w = int(img.shape[1] / s)
    h = int(img.shape[0] / s)
    dimension = (w, h)
    img = cv2.resize(img, dimension, interpolation=cv2.INTER_LANCZOS4)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
    red_hsv = HSV_RED(HSV)
    image_red = cv2.bitwise_and(img,img, mask=red_hsv)
    red = finding_contours(image_red, 4)

    green_hsv = HSV_GREEN(HSV)         
    image_green = cv2.bitwise_and(img,img, mask=green_hsv)
    green = finding_contours(image_green, 1)


    yellow_hsv = HSV_YELLOW(HSV)
    image_yellow = cv2.bitwise_and(img,img, mask=yellow_hsv)
    yellow = finding_contours(image_yellow, 3)

    purple_hsv = HSV_PURPLE(HSV)   
    image_purple = cv2.bitwise_and(img,img, mask=purple_hsv)
    purple = finding_contours(image_purple, 2)

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
