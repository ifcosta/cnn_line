import numpy as np
import cv2


def ohta_otsu(image, *args):
    R = image[:,:,2]
    G = image[:,:,1]
    I1_prime = R-G
    #I1_prime = imp.get_img(image)
    blur1 = cv2.GaussianBlur(I1_prime,(21,21),20)
    th,mask1 = cv2.threshold(blur1,
                            0,255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ~mask1/255
