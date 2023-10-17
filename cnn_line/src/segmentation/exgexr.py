


import numpy as np



def exgexr(image, th = 0.3, *args):
    if(isinstance(th, list)):
        th = th[-1]
    im_to_binarize = image.astype('float')
    
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_+0.000001)
    g = G_/(R_+G_+B_+0.000001)
    b = B_/(R_+G_+B_+0.000001)
    
    excess_red = 1.4*r - g
    excess_green = 2*g - r - b
    
    #a = 2*im_to_binarize[:,:,1] - im_to_binarize[:,:,0] - im_to_binarize[:,:,2]
    #excess_red = 1.4*im_to_binarize[:,:,2] - im_to_binarize[:,:,1]
    eG_eR = excess_green - excess_red
    
    thresh2 = np.where(eG_eR > th, 1.0, 0.0)

    return thresh2
