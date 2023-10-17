#!/usr/bin/python3
import PySimpleGUI as sg
import cv2
import os, argparse, math
from pathlib import Path
import numpy as np

labeling = False

kernel3 = np.full((3,3),1)
kernel5 = np.full((3,3),1)

x_axis = np.linspace(-96, 96, 480)[:, None]
y_axis = np.linspace(-128, 128, 640)[None, :]
arr = np.sqrt(x_axis ** 2 + y_axis ** 2)
arr = cv2.merge((255-arr,255-arr,255-arr))

#plant_mask_A = plant_mask_B = plant_mask_C = np.zeros((480, 640), dtype=np.uint8)

def enhance(img):
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a, b = cv2.split(lab)

  # Applying CLAHE to L-channel
  # feel free to try different values for the limit and grid size:
  clahe = cv2.createCLAHE(clipLimit=values["-clahe_clip-"], tileGridSize=(int(values["-clahe_grid-"]),int(values["-clahe_grid-"])))
  cl = clahe.apply(l_channel)

  # merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv2.merge((cl,a,b))

  # Converting image from LAB Color model to BGR color spcae
  enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  hsv_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv_img)
  h = np.asarray(h*values["-h_val-"],dtype=np.uint8)#1.15/0.95
  s = np.asarray(np.clip(s*values["-s_val-"], a_min = 0, a_max = 255),dtype=np.uint8)
  v = np.asarray(np.clip(v*values["-v_val-"], a_min = 0, a_max = 255),dtype=np.uint8)

  color_plus = cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)

  return color_plus

def keep_largest_components(binary_mask, num_components_to_keep):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    invert_mask = 255 - binary_mask

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create a new binary mask with the X largest contours filled in
    result_mask = np.zeros_like(binary_mask)

    for i in range(min(num_components_to_keep, len(contours))):
        cv2.drawContours(result_mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
    
    result_mask = np.where(invert_mask==255, 0 , result_mask)

    return result_mask

def exgr_mask(image, th = 0.10, *args):
    if(isinstance(th, list)):
        th = th[-1]
    im_to_binarize = image.astype('float')
    
    R_ = im_to_binarize[:,:,2]/np.max(im_to_binarize[:,:,2])
    G_ = im_to_binarize[:,:,1]/np.max(im_to_binarize[:,:,1])
    B_ = im_to_binarize[:,:,0]/np.max(im_to_binarize[:,:,0])
    
    r = R_/(R_+G_+B_+0.000001)
    g = G_/(R_+G_+B_+0.000001)
    b = B_/(R_+G_+B_+0.000001)
    
    excess_red = r*values["-r_val_1-"] - g*values["-g_val_1-"]
    excess_green = g*values["-g_val_2-"] - r*values["-r_val_2-"] - b*values["-b_val_2-"]
    
    #a = 2*im_to_binarize[:,:,1] - im_to_binarize[:,:,0] - im_to_binarize[:,:,2]
    #excess_red = 1.4*im_to_binarize[:,:,2] - im_to_binarize[:,:,1]
    eG_eR = excess_green - excess_red
    
    thresh2 = np.where(eG_eR > th, 1.0, 0.0)
    mask = cv2.normalize(thresh2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return mask

def hsv_mask(image):
    # convert to hsv colorspace
    image = np.asarray(np.clip(arr+image , a_min = 0, a_max = 255),dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower bound and upper bound for Green color
    #lower_bound, upper_bound = np.array([32, 10, 20]),np.array([105, 255, 255])
    lower_bound, upper_bound = np.array([70, 10, 20]),np.array([102, 255, 255])
    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def green_dec(img):
  global arr
  b,g,r = cv2.split(img)
  #x,y,z = ( 4.0 , 0 , 5.0 )
  x,y,z = ( 1.9 , -1.3 , 2.9 )
  #x,y,z = ( 2.4 , 1.4 , 4.1 )
  b2 = np.asarray(np.clip(150 + (x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)
  g2 = np.asarray(np.clip(150 + (x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)
  r2 = np.asarray(np.clip(150 + (x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)

  img = cv2.merge((b2,g2,r2))
  img = np.asarray(np.clip(arr/2.1+img , a_min = 0, a_max = 255),dtype=np.uint8)

  T,thresh = cv2.threshold(img,222,255,cv2.THRESH_BINARY)
  return cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
  #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blue_dec(img):
  # convert to hsv colorspace
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # lower bound and upper bound for Green color
  lower_bound = np.array([0, 0, 210])   
  upper_bound = np.array([180, 255, 255])
  # find the colors within the boundaries
  mask = cv2.inRange(hsv, lower_bound, upper_bound)
  mask[0:640][260:]=0
  #bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  return mask

def rgb_filter(img, rgb):
  global arr
  b,g,r = cv2.split(img)
  (x,y,z) = rgb#( 4.9 , 0.5 , 3.8 )
  b2 = np.asarray(np.clip(70 + (-x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)
  g2 = np.asarray(np.clip(70 + (-x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)
  r2 = np.asarray(np.clip(70 + (-x*g + y*b - z*r), a_min = 0, a_max = 255), dtype=np.uint8)
  img = cv2.merge((b2,g2,r2))
  #img[0:640][0:220][:]=0
  T,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
  return cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

def to_class(mask):
    r,g,b = cv2.split(mask)
    r = np.clip(r,0,1)
    g = np.clip(g,0,1)
    b = np.clip(b,0,1)
    class0 = np.multiply(b,(1-r))
    class1 = np.multiply(r,(1-b))
    class2 = g
    class3 = np.multiply(r,b)
    t = np.dstack(( class0 , class1 , class2 , class3 ))

    tt = t.argmax(axis=2)

    return np.asarray(tt,dtype=np.uint8)

def fit_line(x_point,y_point,predict_list, camera_pos = 'top'):
    coef = np.polyfit(y_point,x_point,deg=1)

    x = []
    for value in predict_list:
        x.append(int(np.round(coef[0]*value + coef[1])))
    x0 = coef[0]*y_point[0] + coef[1]
    x1 = coef[0]*y_point[1] + coef[1]
    if camera_pos == 'top':
        theta = math.atan2(480,(x1-x0)) - np.pi/2
    elif camera_pos == 'bot':
        theta = - math.atan2(480,(x1-x0)) + np.pi/2

    return x,predict_list,theta

def mouse(event, x, y, flags, param): 
  global start, end, labeling
  
  if event == cv2.EVENT_LBUTTONDOWN:
    start = [x, y]
    end = [x, y]
    labeling = True
  elif event == cv2.EVENT_MOUSEMOVE and labeling == True:
    end = [x, y]
  elif event == cv2.EVENT_LBUTTONUP:
    end = [x, y]
    labeling = False


parser = argparse.ArgumentParser(description='Label navigation images.')
print(parser.add_argument('-s', '--start', default=0, metavar='N', type=int, help='Start at image N'))
print(parser.add_argument('-d', '--dir', default='.', metavar='DIR', type=str, help='Read images from directory DIR'))

args = parser.parse_args()

files = []
for path in Path(args.dir).rglob('*.jpg'): #
  files.append(str(path))
files.sort()

output = 'label'
if args.dir != '.':
  output = output + '-' + args.dir.replace(os.sep, '_')
if args.start > 0:
  output = output + '-' + str(args.start)
output = output + '.txt'

if os.path.exists(output):
  print(f'This will OVERWRITE existing output {output}. Hit Ctrl-C to bail or ENTER to continue.')
  input()

o = open(output, 'w')

cv2.namedWindow(winname = 'Labeling - W: next image; S: prev image; Q: quit') 
cv2.setMouseCallback('Labeling - W: next image; S: prev image; Q: quit', mouse)

jumpto = 0

sg.theme("LightGreen")

# Define the window layout
layout = [
    #[sg.Text("Mask Viewer", size=(60, 1), justification="center")],
    [sg.Radio("Base IMG",group_id=555, size=(9, 1), default=False, key="-base_preview-"),
     sg.Radio("Proc IMG",group_id=555, size=(9, 1), default=False, key="-proc_preview-"),
     sg.Radio("RGB Mask",group_id=555, size=(9, 1), default=True,  key="-rgb_preview-"),
     sg.Radio("Cat Mask",group_id=555, size=(9, 1), default=False, key="-cat_preview-"),
     sg.Checkbox("Show Line",size=(9, 1), default=True, key="-LINE-"),
     sg.Checkbox("Bot Camera",size=(9, 1), default=False, key="-BOT-")],

    [sg.Image(filename="", key="-IMAGE-")],

    [sg.Text("Lin Geom.", size=(9, 1), justification="left"),
        sg.Slider((0, 400), 64, 1, orientation="h", size=(30, 10), key="-Top_Size-"),
        sg.Slider((0, 400), 96, 1, orientation="h", size=(30, 10), key="-Bot_Size-")],
    
    [sg.Text("EXGR Tresh", size=(9, 1), justification="left"),
        sg.Slider((0, 1), 0.1, 0.01, orientation="h", size=(61, 10), key="-Tresh-")],
    
    [sg.Text("EX green", size=(9, 1), justification="left"),
        sg.Slider((-3, 4), 1, 0.1, orientation="h", size=(30, 10), key="-r_val_1-"),
        sg.Slider((-3, 4), 1, 0.1, orientation="h", size=(30, 10), key="-g_val_1-")],
    
    [sg.Text("EX red", size=(9, 1), justification="left"),
        sg.Slider((-3, 4), 1, 0.1, orientation="h", size=(19.5, 10), key="-g_val_2-"),
        sg.Slider((-3, 4), 1, 0.1, orientation="h", size=(19.5, 10), key="-r_val_2-"),
        sg.Slider((-3, 4), 1, 0.1, orientation="h", size=(19.5, 10), key="-b_val_2-")],
    
    [sg.Checkbox("",size=(1, 1), default=True, key="-CLAHE-"),
        sg.Text("CLAHE grid", size=(9, 1), justification="left"),
        sg.Slider((1, 15), 5, 1, orientation="h", size=(25.3, 10), key="-clahe_grid-"),
        sg.Text("clip", size=(3, 1), justification="left"),
        sg.Slider((0.5, 2.5), 1.5, 0.1, orientation="h", size=(25.3, 10), key="-clahe_clip-")],

    [sg.Text("HSV Edit", size=(9, 1), justification="left"),
        sg.Slider((0.7, 1.3), 1, 0.01, orientation="h", size=(19.5, 10), key="-h_val-"),
        sg.Slider((0.7, 1.3), 1, 0.01, orientation="h", size=(19.5, 10), key="-s_val-"),
        sg.Slider((0.7, 1.3), 1, 0.01, orientation="h", size=(19.5, 10), key="-v_val-")],
    
    [sg.Checkbox("",size=(1, 1), default=True, key="-BLUR-"),
        sg.Text("Blur", size=(9, 1), justification="left"),
        sg.Slider((3, 21), 7, 2, orientation="h", size=(25.3, 10), key="-blur_val-"),
        sg.Checkbox("Erode",size=(8, 1), default=True, key="-ERODE-"),
        sg.Checkbox("Dilate",size=(8, 1), default=True, key="-DILATE-")],
    
    [sg.Checkbox("",size=(1, 1), default=True, key="-CONTOUR-"),
        sg.Text("Filter Contour", size=(9, 1), justification="left"),
        sg.Slider((1, 10), 7, 2, orientation="h", size=(25.3, 10), key="-contour_val-"),
        sg.Checkbox("Red Priority",size=(9, 1), default=False, key="-RED-")],

    [sg.Checkbox("",size=(1, 1), default=False, key="-SKY-"),
        sg.Text("Sky Mask", size=(9, 1), justification="left"),
        sg.Slider((-4, 5), 1, 0.1, orientation="h", size=(19.5, 10), key="-x_val-"),
        sg.Slider((-4, 5), 1, 0.1, orientation="h", size=(19.5, 10), key="-y_val-"),
        sg.Slider((-4, 5), 1, 0.1, orientation="h", size=(19.5, 10), key="-z_val-")],

    [sg.Button("Exit", size=(10, 1))]
]

# Create the window and show it without the plot
window = sg.Window("Mask Viewer", layout, location=(100, 50))

for i, f in enumerate(files):
  if i < args.start or i < jumpto:
    continue

  print(f'Labeling file {i+1}/{len(files)-1}: {f}')

  start = None
  end = None 
  labeling = False
  done = False
  exit = False
  startline = (0,0)
  endline = (0,0)
  X = Y = [0,0]
  
  img = cv2.imread(f)
  shape = (img.shape[0], img.shape[1])
  black = np.zeros(shape, dtype=np.uint8)
  white = np.full(shape, 255, dtype=np.uint8)
    
  while not done:
    base_img = img.copy()
    show = img.copy()
    event, values = window.read(timeout=20)
    if event == "Exit" or event == sg.WIN_CLOSED:
        exit = True
        done = True
    
    if start is not None:
      startline = start
      endline = end

      if values["-CLAHE-"]:
        img_enh = enhance(base_img)
      else:
        img_enh = base_img

      if values["-BLUR-"]:
        img_enh = cv2.medianBlur(img_enh, int(values["-blur_val-"]))
      
      #MASKS TO BE USED
      plant_mask_A = exgr_mask(img_enh, th = values["-Tresh-"])
      if values ["-ERODE-"]:
        plant_mask_A = cv2.erode(plant_mask_A, kernel5)
      if values ["-DILATE-"]:
        plant_mask_A = cv2.dilate(plant_mask_A, kernel3)
      if values ["-CONTOUR-"]:
        plant_mask_A = keep_largest_components(plant_mask_A, int(values["-contour_val-"]))

      #plant_mask_B = cv2.dilate(cv2.erode(hsv_mask(img_blur),kernel5),kernel3)
      #plant_mask_C = cv2.dilate(cv2.erode(green_dec(img_blur),kernel5),kernel3)

      #soil_mask    = cv2.dilate(cv2.erode(red_dec(img_mblur),kernel5),kernel3)
      #sky_mask     = cv2.dilate(cv2.erode(blue_dec(img_mblur),kernel5),kernel3)

      cv2.line(show, startline, endline, (0, 0, 0), 7)
      pt1 = list(startline)
      pt2 = list(endline)
      if abs(pt1[0]-pt2[0])>20 or abs(pt1[1]-pt2[1])>20:
        X, Y, theta = fit_line([pt1[0], pt2[0]], [pt1[1], pt2[1]],[0,480])

      cv2.line(show, (X[0],Y[0]), (X[1],Y[1]), (0, 255, 255), 2)

      if values ["-BOT-"]:
        pts = np.array([(pt1[0]-int(values["-Top_Size-"]), pt1[1]),
                        (pt1[0]+int(values["-Top_Size-"]), pt1[1]),
                        (X[1]+int(values["-Bot_Size-"]), 480),
                        (X[1]-int(values["-Bot_Size-"]), 480)])
      else:
        pts = np.array([(X[0]-int(values["-Top_Size-"]), 0),
                        (X[0]+int(values["-Top_Size-"]), 0),
                        (X[1]+int(values["-Bot_Size-"]), 480),
                        (X[1]-int(values["-Bot_Size-"]), 480)])
        
      line_image = cv2.fillPoly(black.copy(), [pts], (255))

      #green = np.clip(plant_mask_A + plant_mask_B + plant_mask_C , a_min = 0, a_max = 255)
      green = plant_mask_A
      
      if values ["-RED-"]:
        red = line_image
        green  = np.where(line_image == 0, green, 0)
      else:
        red  = np.where(green == 0, line_image, 0)

      blue = np.where(green == 0, white, 0)
      blue = np.where(line_image == 0, blue, 0)
      
      if values ["-SKY-"]:
        sky_msk = rgb_filter(img_enh,(values["-x_val-"],values["-y_val-"],values["-z_val-"]))
        #cv2.imshow('test',sky_msk)
        red [0:pt1[1], :] = np.where(green[0:pt1[1], :] == 0, red [0:pt1[1], :] + sky_msk[0:pt1[1], :], 0)
        blue[0:pt1[1], :] = np.where(green[0:pt1[1], :] == 0, blue[0:pt1[1], :] + sky_msk[0:pt1[1], :], 0)
        #red [0:pt1[1], :] = np.where(green[0:pt1[1], :] == 0, 255, 0)
        #blue[0:pt1[1], :] = np.where(green[0:pt1[1], :] == 0, 255, 0)

      plant_mask_bgr = cv2.merge((blue, green, red))
      img_composite = cv2.addWeighted(plant_mask_bgr, 0.6, base_img, 0.5, 25)

      plant_mask_cat = to_class(plant_mask_bgr)

      #cv2.imshow('Output', img_composite)
      if values["-base_preview-"]:
        imshow = base_img
      elif values["-proc_preview-"]:
        imshow = img_enh
      elif values["-rgb_preview-"]:
        imshow = img_composite
      elif values["-cat_preview-"]:
        imshow = plant_mask_cat*32
      
      if values ["-LINE-"]:
        imshow = cv2.line(imshow, (X[0],Y[0]), (X[1],Y[1]), (0, 0, 0), 7)
        imshow = cv2.line(imshow, (X[0],Y[0]), (X[1],Y[1]), (0, 255, 255), 2)
    
    if start is None:
      imgbytes = cv2.imencode(".png", base_img)[1].tobytes()
    else:
      imgbytes = cv2.imencode(".png", imshow)[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    # show to the user
    cv2.putText(show,'Drag mouse to label path, then press w; s to skip, q to quit', 
    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow('Labeling - W: next image; S: prev image; Q: quit', show)
    res = cv2.waitKey(1)
    
    if res == 113:
      done = True
      exit = True

    if res == 119:
      if start is None:
        print('Must either label or skip image')
      else:
        print((X[0],Y[0]), (X[1],Y[1]))

        print(f'{i}, {f}, {startline}, {endline}', file=o)
        o.flush()

        out2 = open(f[0:-7]+'lbl.txt', 'w')
        print(X[0], X[1], file=out2)
        out2.flush()
        out2.close()
        
        cv2.imshow('Labeling - W: next image; S: prev image; Q: quit', show)
        #cv2.imshow('Plant A', plant_mask_A)
        #cv2.imshow('Plant B', plant_mask_B)
        #cv2.imshow('Plant C', plant_mask_C)
        #cv2.imshow('sky', sky_mask)
        #cv2.imshow('soil', soil_mask)
        #cv2.imshow('COMP', img_composite)
        #cv2.imshow('mask', plant_mask_bgr)
        #cv2.imshow('class', plant_mask_class*30)

        #cv2.waitKey(500)

        cv2.imwrite(f.replace('img.jpg', 'cat.png' ), plant_mask_cat)
        cv2.imwrite(f.replace('img.jpg', 'mask.png' ), plant_mask_bgr)

        done = True
        startline = (0,0)
        endline = (0,0)
        line = 0

    if res == 115:
      print('Skipped') 
      done = True
      startline = (0,0)
      endline = (0,0)
      line = 0

    if res == 99:
      startline = (0,0)
      endline = (0,0)
      start = None
      end = None
      line = 0
      #show = backup.copy()
      print('label reseted')
      
  if exit:
    break

window.close()
o.close()
cv2.destroyAllWindows()

