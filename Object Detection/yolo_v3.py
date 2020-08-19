import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, ZeroPadding2D, LeakyReLU, UpSampling2D, Add

Batch_norm_epsilent = 1e-05
Batch_norm_decay = 0.9
LeakyReLU_alpha = 0.1
ANCHORS = [(10, 13), (16, 30), (33, 23),
           (30, 61), (62, 45), (59, 119),
           (116, 90), (156, 198), (373, 326)]


def Batch_norm(input_layer, training, data_format):
    if data_format == 'channels_first':
        X = tf.keras.layers.BatchNormalization(axis=1)(input_layer)
    else:
        X = tf.keras.layers.BatchNormalization(3, momentum=Batch_norm_decay, epsilon=Batch_norm_epsilent,
                                               trainable=training)(input_layer)
    return X


def fixed_padding(input_layer, size, data_format):
    total_pad = size - 1
    pad_beg = total_pad // 2
    pad_end = total_pad - pad_beg

    if data_format == 'channels_first':
        X = tf.pad(input_layer, [[0, 0], [0, 0],
                                 [pad_beg, pad_end],
                                 [pad_beg, pad_end]])
    else:
        X = tf.pad(input_layer, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return X


def Conv_pad(input_layer, channels, size, data_format, strides=1):
    if strides > 1:
        input_layer = fixed_padding(input_layer, size, data_format)
    X = Conv2D(channels, size, strides=strides, padding=('same' if strides == 1 else 'valid'), use_bias=False,
               data_format=data_format)(input_layer)
    return X


def DBL_layer(input_layer, channels, size, training, dataformat):
    X = Conv_pad(input_layer, channels, size, dataformat)
    X = Batch_norm(X, training, dataformat)
    output = LeakyReLU(alpha=0.1)(X)
    return output


def Yolo_conv(input_layer, channels, training, data_format):
    X = DBL_layer(input_layer, channels, 1, training, data_format)
    X = DBL_layer(X, channels * 2, 3, training, data_format)
    X = DBL_layer(X, channels, 1, training, data_format)
    X = DBL_layer(X, channels * 2, 3, training, data_format)
    X = DBL_layer(X, channels, 1, training, data_format)
    route = X
    output = DBL_layer(X, channels * 2, 3, training, data_format)
    return route, output


def Yolo_layer(input_layer, anchors, num_classes, imsize, data_format):
    ##### yolo head #####

    n_anchors = len(anchors)
    X = Conv2D(n_anchors * (5 + num_classes), 1, 1, use_bias=True, data_format=data_format)(input_layer)
    shape = tf.shape(X)
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        X = tf.transpose(X, [0, 2, 3, 1])
    X = tf.reshape(X, ([-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + num_classes]))
    strides = (imsize[0] // grid_shape[0], imsize[1] // grid_shape[1])
    box_center, box_shape, confidence, classe_validation = tf.split(X, [2, 2, 1, num_classes], axis=-1)
    X = tf.range(grid_shape[0],) #dtype=tf.float32)
    Y = tf.range(grid_shape[1],) #dtype=tf.float32)
    X_offset, Y_offset = tf.meshgrid(X, Y)
    X_offset = tf.reshape(X_offset, (-1, -1))
    Y_offset = tf.reshape(Y_offset, (-1, -1))
    X_Y_offset = tf.concat([X_offset, Y_offset], axis=-1)
    X_Y_offset = tf.tile(X_Y_offset, [1, n_anchors])
    X_Y_offset = tf.reshape(X_Y_offset, [1, -1, 2])
    box_center = tf.nn.sigmoid(box_center)
    box_center = (box_center + X_Y_offset) * strides
    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shape = tf.exp(box_shape)
    confidence = tf.nn.sigmoid(confidence)
    classe_validation = tf.nn.sigmoid(classe_validation)
    output = tf.concat([box_center, box_shape, confidence, classe_validation], axis=-1)
    return output



def build_boxes(input):
    x_centre, y_centre, w, h, confident, classes = tf.split(input, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = x_centre - w / 2
    top_left_y = y_centre - h / 2
    bottom_right_x = x_centre + w / 2
    bottom_right_y = y_centre + h / 2
    boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confident, classes], axis=-1)
    return boxes


def non_max_supression(input, classes, max_boxes, IOU, confident_min):
    batch = tf.unstack(input)
    box_dict = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confident_min)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)
        for cl in range(classes):
            mask = tf.equal(boxes[:, :5], cl)
            mask_shape = tf.shape(mask)
            if mask_shape.ndims != 0:
                box_clasess = tf.boolean_mask(boxes, mask)
                box_cords, conf_scores = tf.split(box_clasess, [4, 1, -1], axis=-1)
                conf_scores = tf.reshape(conf_scores, [-1])
                indice = tf.image.non_max_suppression(box_cords, conf_scores, max_output_size=max_boxes,
                                                      iou_threshold=IOU)
                box_clasess = tf.gather(box_clasess, indice)
                box_dict[i] = box_clasess[:, :5]

    return box_dict


def ConvBN(input_layer, channels, size, training, data_format, strides=1):
    X = Conv_pad(input_layer, channels, size, data_format, strides)
    X = Batch_norm(X, training, data_format)
    output = LeakyReLU(alpha=LeakyReLU_alpha)(X)
    return output


def darknet_block(input_layer, channels, training, data_format, strides=1):
    input_shortcut = input_layer
    X = ConvBN(input_layer, channels, 1, training, data_format, strides)
    X = ConvBN(X, channels * 2, 3, training, data_format, strides)

    output = Add()([input_shortcut, X])
    return output


def Darknet(input_layer, training, data_format):
    X = ConvBN(input_layer, 32, 3, training, data_format)
    X = ConvBN(X, 64, 3, training, data_format, 2)

    X = darknet_block(X, 32, training, data_format)
    X = ConvBN(X, 128, 3, training, data_format, 2)

    for i in range(2):
        X = darknet_block(X, 64, training, data_format)
    X = ConvBN(X, 256, 3, training, data_format, 2)

    for i in range(8):
        X = darknet_block(X, 128, training, data_format)
    route1 = X
    X = ConvBN(X, 512, 3, training, data_format, 2)

    for i in range(8):
        X = darknet_block(X, 256, training, data_format)
    route2 = X
    X = ConvBN(X, 1024, 3, training, data_format, 2)

    for i in range(4):
        output = darknet_block(X, 512, training, data_format)

    return route1, route2, output


def Yolo_v3_model(model_size, num_classes, training, data_format):
    if not data_format:
        if tf.test.is_built_with_cuda():
            data_format = 'channels_first'
    else:
        data_format = 'channels_last'

    input_layer = Input(model_size)

    if data_format == 'channels_first':
        input_layer = tf.transpose(input_layer, [0, 3, 1, 2])
    X = input_layer / 255.0

    route1, route2, X = Darknet(input_layer, training, data_format)

    route, X = Yolo_conv(X, 512, training, data_format)

    detect1 = Yolo_layer(X, ANCHORS[6:9], num_classes, model_size, data_format)

    X = ConvBN(route, 256, 1, training, data_format)
    if data_format == 'channels_first':
        axis = 1
        up_sample_size(2, 1)
    else:
        axis = 3
        up_sample_size = (2, 2)

    X = UpSampling2D(up_sample_size, data_format=data_format)(X)

    X = tf.concat([X, route2], axis=axis)

    route, X = Yolo_conv(X, 256, training, data_format)

    detect2 = Yolo_layer(X, ANCHORS[3:6], num_classes, model_size, data_format)

    X = ConvBN(route, 128, 1, training, data_format)

    X = UpSampling2D(up_sample_size, data_format=data_format)(X)

    X = tf.concat([X, route1], axis=axis)

    route, X = Yolo_conv(X, 128, training, data_format)

    detect3 = Yolo_layer(X, ANCHORS[0:3], num_classes, model_size, data_format)

    detect = tf.concat([detect1, detect2, detect3], axis=1)

    model = Model(inputs=input_layer, outputs=detect)

    return model
