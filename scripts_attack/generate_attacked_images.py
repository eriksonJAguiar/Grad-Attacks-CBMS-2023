from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, UniversalPerturbation, DeepFool
from art.estimators.classification import KerasClassifier
from art.estimators.classification import TensorFlowV2Classifier

#Utils
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import csv
import math

#Tf imports
from tensorflow.keras.utils import image_dataset_from_directory as load_images
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import time
import argparse
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.math import argmax
import sklearn.metrics as mc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

#arguments
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--database_csv', type=str, required=True, help="database path to collect images, such as chest_xray or OCT")
parser.add_argument('--model_name', type=str, required=True, help="CNN name such as inceptionv3, vgg16, and nasnet")
parser.add_argument('--attack_name', type=str, required=True, help="Attack to run on images, such as FGSM, PGD, or Deep")
# Parse the argument
args = parser.parse_args()

database = args.database_csv.split(".")[0]
model_name = "inceptionv3" if args.model_name is None else args.model_name

batch_size = 32
image_size = (299, 299) if model_name == "inceptionv3" else (224,224)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    
test_csv = pd.read_csv(args.database_csv)

print("Database info: \n")
class_names = test_csv["y"].unique()
print(class_names)
db = database.split("_")[0]
model_trained = load_model(os.path.join("weights","model_{}_{}.hdf5".format(model_name, db)), custom_objects={"f1_score": f1_score, "mcc": mcc})

attack = "PGD" if args.attack_name is None else args.attack_name
nb = len(class_names)
input_shape = (299,299) if model_name ==  "inceptionv3" else (224,224)


classifier = TensorFlowV2Classifier(model=model_trained, nb_classes=nb, input_shape=input_shape, loss_object=CategoricalCrossentropy())

#test = test_csv.sample(100)
test = pd.DataFrame()
s  = int(100/nb)
for c in class_names:
    test = pd.concat([test, test_csv[test_csv["y"] == c].sample(s)])


#image_test_folder = os.path.join(".", args.database, "test_balanced")


datagen = ImageDataGenerator(rescale = 1./255)

test_generator = datagen.flow_from_dataframe(dataframe = test, 
                                                target_size = image_size,
                                                shuffle = False,
                                                x_col = "x",
                                                y_col = class_names,
                                                class_mode = "raw", 
                                                batch_size = batch_size
                                            ) 
# test_generator = datagen.flow_from_directory(
#             image_test_folder,
#             target_size=image_size,
#             batch_size=batch_size,
#             class_mode='categorical'
# )
n = len(test_generator)
images, labels = next(test_generator)
#images = np.concatenate([x for x, y in test_generator], axis=0)
#labels = np.concatenate([y for x, y in test_generator], axis=0)

for i in range(n-1):
    img, lb = next(test_generator)
    images = np.concatenate([images, img], axis=0)
    labels = np.concatenate([labels, lb], axis=0)

for attack in ["FGSM", "PGD", "Deep"]:
    for model in ["vgg16", "resnet50", "inceptionv3"]:
        model_name = model
        for eps in [0.01, 0.05, 0.075, 0.1]:
                print("Build attacks on {} and attack {} with eps: {}".format(model_name, attack, eps))

                attack_method = None
                if attack  == "UAP": 
                    attack_method = UniversalPerturbation(classifier=classifier, eps=eps)
                elif attack == "FGSM": 
                    attack_method = FastGradientMethod(estimator=classifier, eps=eps)
                elif attack == "PGD": 
                        attack_method = ProjectedGradientDescent(estimator=classifier, eps=eps)
                elif attack  == "Deep": 
                    attack_method = DeepFool(classifier=classifier, epsilon=eps)
                            
                if not os.path.exists(os.path.join("{}_{}_{}".format(attack, model_name, database))):
                            os.mkdir(os.path.join("{}_{}_{}".format(attack, model_name, database)))
                            
                if not os.path.exists(os.path.join("{}_{}_{}".format(attack, model_name, database), "{}".format(eps))):
                            os.mkdir(os.path.join("{}_{}_{}".format(attack, model_name, database), "{}".format(eps)))
                                    
                for c in class_names:
                        if not os.path.exists(os.path.join("{}_{}_{}".format(attack, model_name, database), "{}".format(eps), c)):
                            os.mkdir(os.path.join("{}_{}_{}".format(attack, model_name, database),"{}".format(eps), c))
                                            
                if not os.path.exists(os.path.join("{}_{}_{}".format(attack, model_name, database), "{}".format(eps), c)):
                            os.mkdir(os.path.join("{}_{}_{}".format(attack, model_name, database), "{}".format(eps), c))
                        
                #     #build attack
                print("Running attack {} ...".format(attack))
                x_test_adv = attack_method.generate(x=images,y=labels)
                            
                print("Save images...")
                fm = test["x"].to_list()
                print(len(x_test_adv))
                for i, (x,y) in enumerate(zip(x_test_adv, labels)):
                        path_images_base = os.path.join("{}_{}_{}".format(attack, model_name, database),"{}".format(eps), class_names[np.argmax(y)])
                        path_images = os.path.join(path_images_base, "{}.{}".format(fm[i].split("/")[-1].split(".")[0], fm[i].split("/")[-1].split(".")[1]))
                        #path_images = "'Example.jpg"
                        #print("Path images {}".format(path_images))
                        cv2.imwrite(path_images, x*255)
