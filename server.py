import streamlit as st
import json
from keras.models import Model, load_model
import urllib.request
import io
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras import backend
import keras.backend.tensorflow_backend as tb


def preprocess_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(f):
    model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
    model_new = Model(model.input, model.layers[-2].output)
    img = preprocess_img(f)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def predict_caption(model1, photo):
    in_text = "startseq"
    for _ in range(80):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=80, padding="post")
        ypred = model1.predict([photo, sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += " " + word
        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption


@st.cache
def load_model_disk():
    model = load_model("./model.h5")
    return model


def predict(image):
    model = load_model_disk()
    return predict_caption(model, encode_image(image).reshape((1, 2048)))


word_to_idx = {}
idx_to_word = {}

with open("word_to_idx.txt", "r") as f:
    word_to_idx = eval(f.read())
with open("idx_to_word.txt", "r") as f:
    idx_to_word = eval(f.read())

st.title("Image Caption Bot")

selected_image = st.file_uploader("Upload an image", type=["png", "jpg"])

# To fix error generated due to karas 2.3.1
tb._SYMBOLIC_SCOPE.value = True


if selected_image is not None:
    st.image(selected_image, use_column_width=True)
    if st.button("Start Prediction"):
        out = predict(selected_image)
        st.markdown(f"`{out.capitalize()}`")
