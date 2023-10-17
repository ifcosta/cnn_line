import cv2, math
import numpy as np

import tensorflow as tf

from time import time,sleep
from tensorflow import keras
from keras import layers


COLORMAP = np.asarray([[255,0,0],[0,0,255],[0,255,0],[255,0,255]])

#DeepLab
def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias,
    kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear")(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def get_x0(x):
    shape = x.shape

    flatten = layers.Flatten(input_shape=shape)(x)
    hidden_layer = layers.Dense(25, activation='relu', use_bias=True)(flatten)
    hidden_layer = layers.Dense(20, activation='relu', use_bias=True)(hidden_layer)
    output_x0 = layers.Dense(1, activation='sigmoid', use_bias=True, name="x0_out")(hidden_layer)

    return output_x0

def get_x1(x):
    shape = x.shape

    flatten = layers.Flatten(input_shape=shape)(x)
    hidden_layer = layers.Dense(25, activation='relu', use_bias=True)(flatten)
    hidden_layer = layers.Dense(20, activation='relu', use_bias=True)(hidden_layer)
    output_x1 = layers.Dense(1, activation='sigmoid', use_bias=True, name="x1_out")(hidden_layer)

    return output_x1

def DeeplabV3_mobile2(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    MobileNet2 = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=model_input)
    x = MobileNet2.get_layer("block_13_depthwise_relu").output

    x1_out = get_x1(x)
    x0_out = get_x0(x)

    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    input_b = MobileNet2.get_layer("block_2_add").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)

    x = layers.UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", name="mask_output")(x)
    return keras.Model(inputs=model_input, outputs=[model_output, x0_out, x1_out])

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=model_input)
    x = resnet50.get_layer("conv4_block6_2_relu").output
    
    x1_out = get_x1(x)
    x0_out = get_x0(x)

    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)

    x = layers.UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", name="mask_output")(x)
    return keras.Model(inputs=model_input, outputs=[model_output, x0_out, x1_out])

def LineModel(image_size):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=model_input)
    x = resnet50.get_layer("conv4_block6_2_relu").output
    
    x1_out = get_x1(x)
    x0_out = get_x0(x)
    return keras.Model(inputs=model_input, outputs=[ x0_out, x1_out])

#Network Inference
def decode_segmentation_masks(mask, colormap, n_classes):
    shape = (IMAGE_SIZE,IMAGE_SIZE)
    r = np.empty(shape,dtype=np.uint8)
    g = np.empty(shape,dtype=np.uint8)
    b = np.empty(shape,dtype=np.uint8)
        
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def predict(image):
    soft_mask,x0,x1 = model.predict(np.expand_dims((image), axis=0))
    soft_mask = np.squeeze(soft_mask)
    x0 = np.squeeze(x0)
    x1 = np.squeeze(x1)

    prediction_colormap = decode_segmentation_masks(np.argmax(soft_mask, axis=2), COLORMAP, NUM_CLASSES) #2.5ms
        
    return prediction_colormap, soft_mask, x0, x1

def predict_line(image):
    x0,x1 = model.predict(np.expand_dims((image), axis=0))
    x0 = np.squeeze(x0)
    x1 = np.squeeze(x1)
    return x0, x1

def fit_line(x_point,y_point,predict_list):
    coef = np.polyfit(y_point,x_point,deg=1)

    x = []                                     #self.theta = np.arctan2((self.x1_pred - self.x0_pred),img.shape[0])
    for value in predict_list:                 #self.x0_pred = int(self.x0_pred + (self.x0_pred-self.x1_pred))
        x.append(int(np.round(coef[0]*value + coef[1]))) #self.x1_pred = int(self.x1_pred)

    x0 = coef[0]*y_point[0] + coef[1]
    x1 = coef[0]*y_point[1] + coef[1]
    if camera_mode == 'top':
        theta = math.atan2(480,(x0-x1)) - np.pi/2
    elif camera_mode == 'bot':
        theta = - math.atan2(480,(x0-x1)) + np.pi/2

    return x,predict_list,theta

def view(image=None):
    cv2.putText(image,f'FPS: ', (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image,f'Spd: A ',(1, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image,f'infr: ms', (540, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image,f'view: ms', (535, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image,f'ctrl: ms', (540, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imshow('Camera View ('+camera_mode+')- Q: Quit', image)

def pred_mode(frame):
        if camera_mode == 'top':
            img = frame
            img_small = cv2.resize(img,dsize=(IMAGE_SIZE, IMAGE_SIZE))

            mask_pred, soft_pred, x0_pred, x1_pred = predict((img_small/127.5) -1)
            x0_pred = int(x0_pred*img.shape[1]*3 - img.shape[1])
            x1_pred = int(x1_pred*img.shape[1]*3 - img.shape[1])

            x_predic,_,theta = fit_line([x1_pred,x0_pred],[0,480],[480])

        elif camera_mode == 'bot':
            img = frame
            img_small = cv2.resize(img,dsize=(IMAGE_SIZE, IMAGE_SIZE))

            #mask_pred, soft_pred, x0_pred, x1_pred = predict((img_small/127.5) -1)
            x0_pred, x1_pred = predict_line((img_small/127.5) -1); mask_pred = img_small

            x0_pred = int(x0_pred*img.shape[1]*3 - img.shape[1])
            x1_pred = int(x1_pred*img.shape[1]*3 - img.shape[1])

            x_predic,_,theta = fit_line([x1_pred,x0_pred],[0,480],[50])

        mask_pred = cv2.cvtColor(mask_pred, cv2.COLOR_RGB2BGR)
        mask_pred = cv2.resize(mask_pred,(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST)
        pred_comp = cv2.addWeighted(img, 1.0, mask_pred, 0.0, 0)

        #pred_comp = cv2.line(pred_comp,(x_predic[0], 0), (x1_pred, img.shape[0]), (255, 255, 0),1)
        mask_pred = cv2.line(pred_comp,(x0_pred, 0), (x1_pred, mask_pred.shape[0]), (0, 255, 255),2)
        pred_comp = cv2.resize(pred_comp, (1024, 768))
        view(pred_comp)

        return

def crop_and_resize(frame, desired_width, desired_height):
    height, width = frame.shape[:2]
    is_vertical = height > width
    #print(height, width, is_vertical)

    if width == desired_width and height == desired_height:
        # Frame is already in the desired resolution
        return frame
    
    if is_vertical:
        # Rotate the frame counter-clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]

    desired_aspect_ratio = desired_width / desired_height
    current_aspect_ratio = width / height

    if current_aspect_ratio > desired_aspect_ratio:
        # Crop horizontally
        new_width = int(height * desired_aspect_ratio)
        offset = (width - new_width) // 2
        cropped_frame = frame[:, offset:offset + new_width, :]
    else:
        # Crop vertically
        new_height = int(width / desired_aspect_ratio)
        offset = (height - new_height) // 2
        cropped_frame = frame[offset:offset + new_height, :, :]

    # Resize to desired resolution
    resized_frame = cv2.resize(cropped_frame, (desired_width, desired_height))

    return resized_frame

def enhance(img, clip_limit=0.6, h_factor=1, s_factor=1.2, v_factor=1):
    # Convert image from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(5, 5))
    cl = clahe.apply(l)

    # Merge enhanced L-channel with a and b channels
    lab_enhanced = cv2.merge((cl, a, b))

    # Convert enhanced LAB image to RGB color space
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Convert RGB enhanced image to HSV color space
    hsv_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2HSV)

    # Adjust H, S, and V channels in-place
    hsv_enhanced[:, :, 0] = np.asarray(np.clip(hsv_enhanced[:, :, 0] * h_factor, 0, 179), dtype=np.uint8)
    hsv_enhanced[:, :, 1] = np.asarray(np.clip(hsv_enhanced[:, :, 1] * s_factor, 0, 255), dtype=np.uint8)
    hsv_enhanced[:, :, 2] = np.asarray(np.clip(hsv_enhanced[:, :, 2] * v_factor, 0, 255), dtype=np.uint8)

    # Convert HSV enhanced image back to RGB color space
    rgb_plus = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    # Convert RGB enhanced image to BGR color space
    color_plus = cv2.cvtColor(rgb_plus, cv2.COLOR_RGB2BGR)

    return color_plus

IMAGE_SIZE = 512
NUM_CLASSES = 4
camera_mode = 'bot'

# Create a VideoCapture object and read from input file
#cap = cv2.VideoCapture('video2_480p_2.mp4')
cap = cv2.VideoCapture('video_5_bot.mp4')
  
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

#model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
#model.load_weights('resnet_'+camera_mode+'_'+str(IMAGE_SIZE)+'_line100.hdf5')

model = LineModel(image_size=IMAGE_SIZE)
model.load_weights('reLine_'+camera_mode+'_'+str(IMAGE_SIZE)+'p200_x1.hdf5') #os.getcwd()


# Read until video is completed
while(cap.isOpened()):
    start = time()
      
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

    # Display the resulting frame
        frame_2 = crop_and_resize(frame,640,480)
        frame_3 = enhance(frame_2, clip_limit=0.7, h_factor=1.1, s_factor=1.1, v_factor=1)
        pred_mode(frame_2)
        #cv2.imshow('Frame', frame_2)
          
    # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
# Break the loop
    else:
        break

    done = time()

    length = 50 -((done-start)*1000)
    if (length) < 0:
        #print(-length, 'over')
        pass
    else:
        #print(length, 'left')
        sleep(length/1000)

  
# When everything done, release
# the video capture object
cap.release()
  
# Closes all the frames
cv2.destroyAllWindows()