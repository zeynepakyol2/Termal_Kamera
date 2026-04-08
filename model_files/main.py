import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model



def load_and_preprocess_wound(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img


def load_and_preprocess_risk(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def load_and_preprocess_stage(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():

    IMG_SIZE = (224, 224)
    wound_model=load_model("yara_var_mi_yok_mu1_model_best.h5")
    risk_model=load_model("wound_risk_model_best.h5")
    stage_model=load_model("wound_stage_best.h5")
    
    classes1=["Derin Doku Hasarı","Evre 1","Evre 2","Evre 3","Evre 4","Evrelendirilemeyen"]
    classes2=["Goreceli_Risk", "Yuksek_Risk", "Cok_Yuksek_Risk"]
    threshold=0.50
    img="001.21.GLUTEAL.EVRE1.-0,5.jpeg"

    wound_img=load_and_preprocess_wound(img)
    pred=wound_model.predict(wound_img)[0]

    if pred[0]>=threshold:
        print("Yara Var → Evre modeline geçiliyor")
        stage_img=load_and_preprocess_stage(img)
        pred=stage_model.predict(stage_img)
        print(pred)
        pred_class=np.argmax(pred).astype(int)
        print("Evre Tahmini:", classes1[pred_class])

    else:
        print("Yara Yok → Risk modeline geçiliyor")
        risk_img=load_and_preprocess_risk(img)
        pred=risk_model.predict(risk_img)
        print(pred)
        pred_class = np.argmax(pred).astype(int)
        print("Risk Tahmini:", classes2[pred_class])



if __name__ == "__main__":
    main()
