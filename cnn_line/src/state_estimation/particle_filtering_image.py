#!/usr/bin/env python

import numpy as np
import cv2
import math

# considering a particle = [h, l, rw, rd]
# I think it is [h, l, rd, rw]

def transition(particles, u):
    dx = u[0]
    dh = u[1]
    #particles[:,0] = np.clip(particles[:,0] + dh + np.random.normal(loc=0.0, scale=0.01,size=particles[:,0].shape), - 30 * np.pi/180, 30 * np.pi/180)
    #particles[:,1] = np.clip(particles[:,1] + dx * np.sin(particles[:,0]) + np.random.normal(loc=0.0, scale=0.1,size=particles[:,1].shape), lb[1], ub[1])
    #particles[:,2] = np.clip(particles[:,2] + np.random.normal(loc=0.0, scale=10,size=particles[:,2].shape), lb[2], ub[2])
    #particles[:,3] = np.clip(particles[:,3] + np.random.normal(loc=0.0, scale=10,size=particles[:,3].shape), lb[3], ub[3])
    particles[:,0] = np.clip(particles[:,0] + dh + np.random.normal(loc=0.0, scale=0.01,size=particles[:,0].shape), -np.inf, np.inf)#- 30 * np.pi/180, 30 * np.pi/180)
    particles[:,1] = particles[:,1] + dx * np.sin(particles[:,0]) + np.random.normal(loc=0.0, scale=0.5,size=particles[:,1].shape)
    particles[:,2] = np.clip(particles[:,2] + np.random.normal(loc=0.0, scale=10,size=particles[:,2].shape), 0, np.inf)
    particles[:,3] = np.clip(particles[:,3] + np.random.normal(loc=0.0, scale=10,size=particles[:,3].shape), 0, particles[:, 2] - 1)
    return particles


def create_masks(particles, observation, plot = 0):
    masks = []
    shape = observation.shape
    value = 200
    shape = (shape[0], shape[1] + value)
    for particle in particles:
        aux = np.zeros(shape)
        central_x = shape[1]//2
        mean_l = central_x - particle[2]//2 + particle[1]
        mean_r = central_x + particle[2]//2 + particle[1]
        left_l = np.clip(mean_l - particle[3]//2, 0, shape[1])
        right_l = np.clip(mean_l + particle[3]//2, 0, shape[1])
        left_r = np.clip(mean_r - particle[3]//2, 0, shape[1])
        right_r = np.clip(mean_r + particle[3]//2, 0, shape[1])

        aux[:,int(left_l):int(right_l)] = 1
        aux[:,int(left_r):int(right_r)] = 1
        #if(plot):
        #    aux = 1-aux

        #aux = skew_image(aux, int(particle[0] * 4 * 180./np.pi))
        aux = skew_image(aux, (particle[0] * 4 * 180./np.pi))


        aux = aux[:, value/2:shape[1]-value/2]

        masks.append(aux)
    return np.array(masks)

def skew_image(image, value = 0, constants = [120, 280]):

    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]

    value = value * (IMAGE_W + IMAGE_H)/400.
    #aa = 100./400*IMAGE_W + value 
    #bb = 300./400*IMAGE_W + value 
    
    aa = np.array(constants[0], dtype=np.float32)/400*IMAGE_W + value 
    bb = np.array(constants[1], dtype=np.float32)/400*IMAGE_W + value 

    src = np.float32([[40./400 * IMAGE_W, 0], [360./400 * IMAGE_W, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
    dst = np.float32([[aa, 0], [bb, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])

    M = cv2.getPerspectiveTransform(src, dst) 
    Minv = cv2.getPerspectiveTransform(dst, src) 

    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    
    return warped_img

def fitness(particles, observation):
    pred = create_masks(particles, observation)
    observation = observation/(np.max(observation) + 1.)
    observation = observation[None, :].astype(np.float32)
    intersection = observation * pred
    notObservation = 1 - observation
    union = observation + (notObservation * pred)

    result = (np.sum(intersection, axis=(-1,-2)) + 1.e-100) / (np.sum(union, axis=(-1,-2)) + 1.e-100)

    return result


'''
def exg(im_to_binarize):
    im_to_binarize = im_to_binarize.astype(np.float)
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_)
    g = G_/(R_+G_+B_)
    b = B_/(R_+G_+B_)
    
    excess_red = 1.4*r - g
    excess_green = 2*g - r - b
    return excess_green


def exg_th(img, th = [0., 0.5]):
    a = exg(img)
    b = np.zeros(shape = a.shape)
    b[a<th[0]] = 0
    b[(a>=th[0]) & (a < th[1])] = (a[(a>=th[0]) & (a < th[1])] - th[0])/(th[1] - th[0])
    b[a >= th[1]] = 1
    return b
'''




def blend_images(rgb, mask):
    aux = cv2.merge([mask * 255, np.zeros(mask.shape), np.zeros(mask.shape)])
    aux = aux.astype(np.uint8)

    return cv2.addWeighted(rgb, 1, aux, 5, 0.0)

def draw_arrow(image, l, h, scale = 1):
    pt1 = (int(image.shape[1]/2 + l*scale), image.shape[0])
    pt2 = (int(image.shape[1]/2 + l*scale + image.shape[1]/2 * math.tan(h)), image.shape[0]/2)

    cv2.arrowedLine(image, pt1, pt2,(0,0,255), 4)
    return image


def blend_images_big(rgb,mask):
    aux = cv2.merge([mask * 255, np.zeros(mask.shape), np.zeros(mask.shape)])
    aux = cv2.resize(aux, (rgb.shape[1],rgb.shape[0]))
    aux = aux.astype(np.uint8)

    return cv2.addWeighted(rgb, 1, aux, 5, 0.0)
