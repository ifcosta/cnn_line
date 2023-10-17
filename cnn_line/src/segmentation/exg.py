import numpy as np

def exg(im_to_binarize):
    im_to_binarize = im_to_binarize.astype(np.float32)      #np.float deprecated -> np.float32 (mar/2023)
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_+0.00001)
    g = G_/(R_+G_+B_+0.00001)
    b = B_/(R_+G_+B_+0.00001)
    
    excess_red = 1.4*r - g
    excess_green = 2*g - r - b
    return excess_green


def exg_th(img, th = [0.1, 0.3], *args):
    a = exg(img)
    b = np.zeros(shape = a.shape)
    b[a<th[0]] = 0
    b[(a>=th[0]) & (a < th[1])] = (a[(a>=th[0]) & (a < th[1])] - th[0])/(th[1] - th[0])
    b[a >= th[1]] = 1
    return b


def hard_exg_th(img, th = 0.1, *args):
    if(isinstance(th, list)):
        th = th[-1]
    return exg_th(img, th=[th,th])
