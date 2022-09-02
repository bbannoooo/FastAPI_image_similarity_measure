from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
import pickle
import pandas as pd
from difflib import SequenceMatcher
import os
from PIL import Image
from io import BytesIO
size = (256, 256)

async def embedding(img):
    
    img = await img.read()
    img = Image.open(BytesIO(img))
    img.resize(size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
    


async def ml(files):


    # 모델로드
    autoencoder = load_model('image_autoencoder_2.h5', compile=False)
    latent_space_model = Model(autoencoder.input, autoencoder.get_layer('latent_space').output)
    
    # 메타데이터 정의
    input_img = files[0]
    embedded_input_img = embedding(input_img)
    input_pred = latent_space_model.predict(embedded_input_img)
    input_pred = np.resize(input_pred, (16))    
    
    compare_img = files[1:]
    
    compare_preds = []
    for file in compare_img:
        embedded_compare_img = embedding(file)
        compare_pred = latent_space_model.predict(embedded_compare_img)
        compare_pred = np.resize(input_pred, (16))
        compare_preds.append(compare_pred)


    # 이미지유사도 함수 계산 정의
    def eucledian_distance(x,y):
        eucl_dist = np.linalg.norm(x - y)
        return eucl_dist

    print(input_pred)
