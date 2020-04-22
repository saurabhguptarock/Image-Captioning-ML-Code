import streamlit as st
import json
from keras.models import Model, load_model
import urllib.request
import io
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_new = Model(model.input, model.layers[-2].output)


def preprocess_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(f):
    img = preprocess_img(f)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


word_to_idx = {}
idx_to_word = {}

with open("word_to_idx.txt", "r") as f:
    word_to_idx = eval(f.read())
with open("idx_to_word.txt", "r") as f:
    idx_to_word = eval(f.read())
