import numpy as np
import cv2

from copy import deepcopy

def normalize(data):
    data_n = np.ma.array(data, mask=np.logical_or(np.logical_or(np.isnan(data),data < 1000),data > 1800))
    data_n = (data_n.astype(np.float32) - np.min(data_n)) / (np.max(data_n) - np.min(data_n)) * 255
    return data_n

def correct_image(image):
	image = np.where(image < 1000, np.nan, image)
	image = np.where(image > 1800, np.nan, image)
	return image

def clahe_otsu(img, *args):
    #global imagem, imagempp, imagem_corrigida
    #img = cv2.medianBlur(img,15)
    
    #imagem_corrigida = deepcopy(image)
	
    '''
    imagem_corrigida.setflags(write=1)
    where_are_NaNs = np.isnan(imagem_corrigida)
    imagem_corrigida[where_are_NaNs] = 3.0
    '''
	
    #yf, xf = img.shape
    #x = int(0.25*xf)
	
    #img = img[200:yf-100, x:xf-x]
	
	
    #img_c = correct_image(img)
    img_n = normalize(img).astype(np.uint8)
    #img = cv2.medianBlur(img,3)
    #img = img[int(0.2*yf):int(yf*0.8),x:int(xf*0.8)]

    #cv2.imshow('name',img)
    #cv2.waitKey(2)
    #print(np.unique(img))
    #cv2.imshow('sad', img)
    #cv2.waitKey(1)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    raise 'asd'


	



    img = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(1,img_n.shape[1])).apply(img_n)
    #cv2.imshow('clahe', img)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    raise 'asd'


    #imagempp = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,10)
    
    _, imagempp = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inverted_image = ~imagempp
    inverted_image[img_n.mask] = 0
    #inverted_image[img == np.nan] = 127
    print(np.where(img == np.nan))
    #cv2.imshow('seg', imagempp)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
    #    raise 'asd'


    #img[img <= 40] = 0
    #img[img > 40] = 1
    #imagempp=img
    #_, imagempp = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    #print(np.unique(imagempp))
    #print(imagempp.shape)
    return inverted_image
