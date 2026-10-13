"""
Model architecture — must match the notebook's build_organ_model /
cbam_block code exactly, since we load trained weights into this graph.
The only change from the notebook is:
  1. output_shape added to both Lambda layers (Keras 3 requires this;
     the notebook's original version works fine in Colab because Keras
     could still trace it there, but a fresh reload of just the config
     cannot infer it).
  2. weights=None on MobileNetV3Large, since we load our own trained
     weights over the whole model anyway and don't need ImageNet
     weights downloaded at startup.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large

IMG_SIZE = 224


def channel_attention_block(x, ratio=8):
    channels = x.shape[-1]
    gap = layers.GlobalAveragePooling2D()(x)
    fc1 = layers.Dense(channels // ratio, activation='relu', use_bias=False)(gap)
    fc2 = layers.Dense(channels, activation='sigmoid', use_bias=False)(fc1)
    fc2 = layers.Reshape((1, 1, channels))(fc2)
    return layers.Multiply()([x, fc2])


def spatial_attention_block(x):
    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        output_shape=lambda s: s[:-1] + (1,),
    )(x)
    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        output_shape=lambda s: s[:-1] + (1,),
    )(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attn_map = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)
    return layers.Multiply()([x, attn_map])


def cbam_block(x, ratio=8):
    return spatial_attention_block(channel_attention_block(x, ratio))


def build_organ_model(num_classes, model_name, dropout=0.4, l2=1e-4):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    backbone = MobileNetV3Large(include_top=False, weights=None, input_tensor=inputs)
    x = cbam_block(backbone.output, ratio=8)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return Model(inputs=inputs, outputs=outputs, name=model_name)