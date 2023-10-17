#!/usr/bin/env python
import os, sys, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}
sys.path.append(os.path.dirname(__file__))

import rospy, math, argparse
import numpy as np

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D

from deep_learning.models import DeeplabV3Plus,LineModel,DeeplabV3_mobile2

#preprocess = False
#net_type = 'resnet'
#camera_pos = 'top'
#net_name = 'resnet_top_512p.hdf5'
#NUM_CLASSES = 3
#IMAGE_SIZE  = 512
#pubrate = 10
#input_topic = '/nav_front/image_raw/compressed'

COLORMAP  = np.asarray([[255,0,0],[0,0,255],[0,255,0],[255,0,255]])
publish = False
cv_image = None
autonomous_mode = False

def image_compressed_callback(msg):
    global cv_image, publish, autonomous_mode
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #cv_image = cv2.cvtColor(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    publish = True
    autonomous_mode = rospy.get_param('/autonomous_mode')

def enhance(img, clip_limit=1.2, h_factor=1.0, s_factor=1.15, v_factor=1.05):
    r_factor = 1 + 0.08
    g_factor = 1 + 0.09
    b_factor = 1 + 0.07

    # Convert image from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(3, 3))
    cl = clahe.apply(l)

    # Merge enhanced L-channel with a and b channels
    lab_enhanced = cv2.merge((cl, a, b))

    # Convert enhanced LAB image to RGB color space
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Convert RGB enhanced image to HSV color space
    hsv_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2HSV)
    #hsv_enhanced = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjust H, S, and V channels in-place
    hsv_enhanced[:, :, 0] = np.asarray(np.clip(hsv_enhanced[:, :, 0] * h_factor, 0, 179), dtype=np.uint8)
    hsv_enhanced[:, :, 1] = np.asarray(np.clip(hsv_enhanced[:, :, 1] * s_factor, 0, 255), dtype=np.uint8)
    hsv_enhanced[:, :, 2] = np.asarray(np.clip(hsv_enhanced[:, :, 2] * v_factor, 0, 255), dtype=np.uint8)

    # Convert HSV enhanced image back to RGB color space
    rgb_plus = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    r,g,b = cv2.split(rgb_plus)
    r = np.asarray(np.clip(r * r_factor, 0, 255), dtype=np.uint8)
    g = np.asarray(np.clip(g * g_factor, 0, 255), dtype=np.uint8)
    b = np.asarray(np.clip(b * b_factor, 0, 255), dtype=np.uint8)
    rgb_plus2 = cv2.merge((r, g, b))

    # Convert RGB enhanced image to BGR color space
    color_plus = cv2.cvtColor(rgb_plus2, cv2.COLOR_RGB2BGR)

    return color_plus

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

    prediction_colormap = decode_segmentation_masks(np.argmax(soft_mask, axis=2), COLORMAP, NUM_CLASSES) #2.5ms
        
    return prediction_colormap, soft_mask, np.squeeze(x0), np.squeeze(x1)

def predict_line(image):
    x0,x1 = model.predict(np.expand_dims((image), axis=0))
    return np.squeeze(x0), np.squeeze(x1)

def inference_pipeline(img, mode = 'resnet', camera_pos = 'top', image_size=256):
    """
    Perform an inference pipeline on the input image.

    Args:
        img (numpy.ndarray): Input image.
        mode (str): Mode of the neural network ('resnet', 'mobile', or 'linnet').
        camera_pos (str): Camera position ('top' or 'bot').
        image_size (int): Size of the input image for inference.

    Returns:
        pred_comp (numpy.ndarray): Processed composite image.
        output_B (numpy.ndarray): Processed image.
        X (float): Predicted X value normalized to the range [-1, 1].
        theta (float): Calculated theta value for the line in radians.
        x0_pred (int): Predicted x0 value, integer [-640, 1280].
        x1_pred (int): Predicted x1 value, integer [-640, 1280].
    """
    if mode == 'resnet' or mode == 'mobile':
        # Resize input image for neural network input
        output_A = cv2.resize(img,dsize=(image_size, image_size))

        # Perform prediction using the neural network
        output_B, soft_pred, x0_pred, x1_pred = predict((output_A/127.5) -1)

        # Adjust predicted x0 and x1 values
        # The predicted values goes from [0 1] and is mapped to [-640 1280] 
        x0_pred = int(x0_pred*img.shape[1]*3 - img.shape[1])
        x1_pred = int(x1_pred*img.shape[1]*3 - img.shape[1])

        # Create a composite image of input and predicted output
        pred_comp = cv2.addWeighted(cv2.cvtColor(output_A, cv2.COLOR_RGB2BGR), 0.7, cv2.cvtColor(output_B, cv2.COLOR_RGB2BGR), 0.3, 10)
        pred_comp = cv2.resize(pred_comp, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST)

        # Resize predicted output for display
        output_B  = cv2.resize(cv2.cvtColor(output_B, cv2.COLOR_RGB2BGR), (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST)

    elif mode == 'linnet':
        # Resize input image for neural network input
        output_A = cv2.resize(img,dsize=(image_size, image_size))

        # Perform prediction using the line detection neural network
        x0_pred, x1_pred = predict_line((output_A/127.5) -1)

        # Adjust predicted x0 and x1 values
        # The predicted values goes from [0 1] and is mapped to [-640 1280] 
        x0_pred = int(x0_pred*img.shape[1]*3 - img.shape[1])
        x1_pred = int(x1_pred*img.shape[1]*3 - img.shape[1])

        # Create a composite image as the output
        pred_comp = cv2.resize(output_A, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_NEAREST)

    y0,y1 = 0,480
    
    # Fit a line to the predicted x0 and x1 values
    x_predic,_,theta = fit_line([x0_pred,x1_pred],[y0,y1],[0,420,480],camera_pos=camera_pos)
    x0_pred = x_predic[0]
    xh_pred = x_predic[1]
    x1_pred = x_predic[2]

    # Draw the line on the composite image for visualization
    pred_comp = cv2.line(pred_comp,(x0_pred, 0), (x1_pred, img.shape[0]), (0,   0,   0),6)
    pred_comp = cv2.line(pred_comp,(x0_pred, 0), (x1_pred, img.shape[0]), (0, 255, 255),2)

    # Calculate normalized final X value
    X = (x_predic[0]/(img.shape[1]/2))-1

    # Adjust X value if camera position is 'bot'
    # The botom camera use the point closest to the botom of the image[X1], while the top view uses the top point[X0]

    if camera_pos == 'bot':
        #pred_comp = cv2.line(pred_comp,(xh_pred, 0), (x1_pred, img.shape[0]), (255, 255, 0),2)
        #x_predic,_,theta = fit_line([xh_pred,x1_pred],[0,480],[0],camera_pos=camera_pos)
        X = (x_predic[1]/(img.shape[1]/2))-1
    
    # Create an output image with the predicted line, this is treated as the result mask obtained from the other model
    if mode == 'linnet':
        output_B = cv2.line(np.full((cv_image.shape[0], cv_image.shape[1],1),0, np.uint8), (x0_pred, 0), (x1_pred, img.shape[0]), (255),6)

    return pred_comp, output_B, X, theta, x0_pred, x1_pred

def data_publish(base_img, proc_img, proc_msk, X, theta, X0, X1):
    global publish

    # Encode the images in JPEG format
    # Compression range goes from 0 to 100, where 100 is the best quality with highest bandwidth usage
    _, img_data = cv2.imencode('.jpg', base_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    img_msg.header.stamp = rospy.Time.now()
    img_msg.data = img_data.tobytes()
    
    _, msk_data = cv2.imencode('.jpg', proc_msk, [cv2.IMWRITE_JPEG_QUALITY, 60])
    msk_msg.header.stamp = rospy.Time.now()
    msk_msg.data = msk_data.tobytes()

    _, cmp_data = cv2.imencode('.jpg', proc_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    cmp_msg.header.stamp = rospy.Time.now()
    cmp_msg.data = cmp_data.tobytes()

    # Publish the compressed images
    img_pub.publish(img_msg)
    msk_pub.publish(msk_msg)
    cmp_pub.publish(cmp_msg)

    # Publish the updated X and line messages
    X_msg.x = X0
    X_msg.y = X1
    lin_msg.x = X
    lin_msg.theta = theta

    pub_X.publish(X_msg)
    pub_lin.publish(lin_msg)

def line_detection():
    while not rospy.is_shutdown():
        # Enables processing while in autonomous mode (autonomous mode is a ros parameter True/False)
        while autonomous_mode == True:
            # Capture the current image from the camera feed
            base_img = cv_image

            # Check if the model exists and there is a image to publish
            if model is not None and publish and base_img is not None:
                if preprocess:
                    # Preprocess the base image if specified
                    # Applies CLAHE algorithm, adds saturation and boost the green
                    base_img = enhance(base_img)
                
                # Perform inference on the image and retrieve the processed line
                proc_img, proc_msk, X, theta, X0, X1 = inference_pipeline(base_img, mode = net_type, camera_pos = camera_pos, image_size=IMAGE_SIZE)

                # Publish the line information
                data_publish(base_img, proc_img, proc_msk, X, theta, X0, X1)

                # Display debug information if debug_mode is enabled
                if debug_mode:
                # Display the original, processed image, and mask for debugging with line X position
                    cv2.putText(proc_img,str(X0) + '   ' + str(X1), (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('output_0',base_img)
                    cv2.imshow('output_1',proc_img)
                    cv2.imshow('output_2',proc_msk)
                    cv2.waitKey(1)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('row_controller')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess',        action='store_true', default=False)
    parser.add_argument('--net_type',          type=str,            nargs='?',default='resnet')
    parser.add_argument('--camera_pos',        type=str,            nargs='?',default='top')
    parser.add_argument('--net_name',          type=str,            nargs='?',default='resnet_top_512p.hdf5')
    parser.add_argument('--num_classes',       type=int,            nargs='?',default=3)
    parser.add_argument('--image_size',        type=int,            nargs='?',default=512)
    parser.add_argument('--publish_frequency', type=float,          nargs='?',default=10)
    parser.add_argument('--input_topic',       type=str,            nargs='?',default='/nav_front/image_raw/compressed')
    parser.add_argument('--debug_mode',        action='store_true',default=False)
    parser.add_argument('__name', type=str,  nargs='?',default='')
    parser.add_argument('__log', type=str,  nargs='?',default='')
    args = parser.parse_args()

    # Extract parsed arguments for configuration
    preprocess  = args.preprocess
    net_type    = args.net_type
    camera_pos  = args.camera_pos
    net_name    = args.net_name
    NUM_CLASSES = args.num_classes
    IMAGE_SIZE  = args.image_size
    pubrate     = args.publish_frequency
    input_topic = args.input_topic
    debug_mode  = args.debug_mode

    # Define topics for communication
    topics = {
        'image_1': input_topic,
        'img_pub': '/line/compressed_image',
        'msk_pub': '/line/compressed_mask',
        'cmp_pub': '/line/compressed_composite',
        'line'   : '/line/x_theta',
        'X_value': '/line/x0_x1',
    }

    # Subscribe to the input image topic
    rospy.Subscriber(topics['image_1'] , CompressedImage , image_compressed_callback)

    # Initialize publishers
    pub_X     = rospy.Publisher(topics['X_value'], Pose2D,   queue_size=10)
    pub_lin   = rospy.Publisher(topics['line']   , Pose2D,   queue_size=10)
    img_pub   = rospy.Publisher(topics['img_pub'], CompressedImage, queue_size=10)
    msk_pub   = rospy.Publisher(topics['msk_pub'], CompressedImage, queue_size=10)
    cmp_pub   = rospy.Publisher(topics['cmp_pub'], CompressedImage, queue_size=10)

    # Initialize message objects
    X_msg   = Pose2D()
    lin_msg = Pose2D()
    img_msg = CompressedImage(); img_msg.format = "jpeg"
    msk_msg = CompressedImage(); msk_msg.format = "jpeg"
    cmp_msg = CompressedImage(); cmp_msg.format = "jpeg"

    # Create a blank image for future use
    blank_img = np.full((640, 480,3),0, np.uint8)

    rospy.loginfo('loading model')

    # Load the specified neural network model based on net_type parameter
    if   net_type == 'resnet':
        model = DeeplabV3Plus(image_size = IMAGE_SIZE, num_classes = NUM_CLASSES)
        model.load_weights(os.path.dirname(__file__)+'/'+'net_weights'+'/' + net_name)
        
    elif net_type == 'mobile':
        model = DeeplabV3_mobile2(image_size = IMAGE_SIZE, num_classes = NUM_CLASSES)
        model.load_weights(os.path.dirname(__file__)+'/'+'net_weights'+'/' + net_name)

    elif net_type == 'linnet':
        model = LineModel(image_size=IMAGE_SIZE)
        model.load_weights(os.path.dirname(__file__)+'/'+'net_weights'+'/' + net_name)
    
    rospy.loginfo('model loaded')
    
    # Set the loop rate for publishing
    rate = rospy.Rate(pubrate)

    # Start the line_detection function
    try:
        line_detection()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass