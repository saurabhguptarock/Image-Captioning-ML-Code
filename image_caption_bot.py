import urllib.request
import io
from PIL import Image
import collections
import json
import pandas as pd
import numpy as np
import keras
import re
import nltk
import string
from nltk.corpus import stopwords
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, LSTM, Embedding
from keras.layers.merge import add


def readTextFile(path: str):
    with open(path) as f:
        captions = f.read()
    return captions


# with open("./flickr30k_images/results.csv", encoding="utf8") as f:
#     file = f.read()
#     with open("./flickr30k_images/data.txt", "w") as w:
#         w.write(file)

captions = readTextFile("./flickr30k_images/data.txt")
captions = captions.split("\n")

descriptions = {}
for x in captions:
    img, comment, data = x.split("|")
    img_name = img.split(".")[0]
    if (descriptions.get(img_name)) is None:
        descriptions[img_name] = []
    descriptions[img_name].append(data)


def clean_text(sentence: str):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+", " ", sentence)
    sentence = sentence.split()
    sentence = " ".join(sentence)
    return sentence


for key, caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])

with open("descriptions.txt", "w") as f:
    f.write(str(descriptions))

IMG_PATH = "./flickr30k_images/flickr30k_images/"

descriptions = None
with open("descriptions.txt", "r") as f:
    descriptions = f.read()

descriptions = json.loads(descriptions.replace("'", '"'))

vocab = set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

total_words = []
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

counter = collections.Counter(total_words)
freq_cnt = dict(counter)

sorted_freq_cnt = sorted(freq_cnt.items(), reverse=True, key=lambda x: x[1])

threshold = 10
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1] > threshold]
total_words = [x[0] for x in sorted_freq_cnt]

train = list(descriptions.keys())

train_descriptions = {}
for img_id in train:
    train_descriptions[img_id] = []
    for cap in descriptions[img_id]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[img_id].append(cap_to_append)

model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_new = Model(model.input, model.layers[-2].output)


def preprocess_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


encoding_train = {}

# for ix, img_id in enumerate(train):
#     img_path = IMG_PATH + img_id + ".jpg"
#     encoding_train[img_id] = encode_image(img_path)

#     if ix % 100 == 0:
#         print(f"Encoding in Progress Time step {ix}")


# with open("encoding_train_features.pkl", "wb") as f:
#     pickle.dump(encoding_train, f)

with open("encoding_train_features.pkl", "rb") as f:
    encoding_train = pickle.load(f)


word_to_idx = {}
idx_to_word = {}

for i, word in enumerate(total_words):
    word_to_idx[word] = i + 1
    idx_to_word[i + 1] = word

idx_to_word[5137] = "startseq"
idx_to_word[5138] = "endseq"
word_to_idx["startseq"] = 5137
word_to_idx["endseq"] = 5138
vocab_size = len(word_to_idx) + 1

max_len = 0
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len = max(max_len, len(cap.split()))

# print(train_descriptions.keys())


def data_generator(
    train_descriptions, encoding_train, word_to_idx, max_len, batch_size
):
    x1, x2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in train_descriptions.items():
            n += 1
            photo = encoding_train[key]
            for desc in desc_list:
                seq = [
                    word_to_idx[word] for word in desc.split() if word in word_to_idx
                ]
                for i in range(1, len(seq)):
                    xi = seq[:i]
                    yi = seq[i]

                    xi = pad_sequences([xi], maxlen=max_len, value=0, padding="post")[0]
                    yi = to_categorical([yi], num_classes=vocab_size)[0]
                    x1.append(photo)
                    x2.append(xi)
                    y.append(yi)
                if n == batch_size:
                    yield [[np.array(x1), np.array(x2)], np.array(y)]
                    x1, x2, y = [], [], []
                    n = 0


f = open("./glove.6B.50d.txt", encoding="utf8")

embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    word_embedding = np.array(values[1:], dtype="float")
    embedding_index[word] = word_embedding
f.close()


def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            matrix[idx] = embedding_vector
    return matrix


embedding_matrix = get_embedding_matrix()

input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.5)(input_img_features)
inp_img2 = Dense(256, activation="relu")(inp_img1)

input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(
    input_captions
)
inp_cap2 = Dropout(0.5)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = add([inp_img2, inp_cap3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

model = Model(inputs=[input_img_features, input_captions], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss="categorical_crossentropy", optimizer="adam")


epochs = 20
batch_size = 3
steps = len(train_descriptions)  # batch_size


def train_batch():
    for i in range(epochs):
        generator = data_generator(
            train_descriptions, encoding_train, word_to_idx, max_len, batch_size
        )
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save("model_" + str(i + 1) + ".h5")


train_batch()


def predict_caption(photo):
    in_text = "startseq"
    for _ in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="post")
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += " " + word
        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption
