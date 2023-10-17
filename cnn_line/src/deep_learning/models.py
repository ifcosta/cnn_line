
import tensorflow as tf
from tensorflow import keras
from keras import layers

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