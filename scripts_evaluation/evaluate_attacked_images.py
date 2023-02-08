from keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory as load_images
import os
import argparse
import time
import csv
import numpy as np
from keras import backend as K
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input



parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--database', type=str, required=True, help="str name of database: chest_xaray or OCT")
parser.add_argument('--attack', type=str, required=True, help="attack: FGSM, PGD, or Deep")
parser.add_argument('--model_name', type=str, required=True, help="CNN name such as: inceptionv3, vgg16, or nasnet")
parser.add_argument('--eps', type=str, required=True, help="Epsilon value [0, 1]")
# Parse the argument
args = parser.parse_args()


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

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

image_size = (299, 299) if args.model_name == "inceptionv3" else (224,224)
base_path = os.path.join(args.database, "test") if args.eps == "0" else os.path.join("{}_{}_{}".format(args.attack, args.model_name, args.database), args.eps) 
file_results = "results_test_{}_{}_{}.csv".format(args.model_name, args.database, args.eps)

header = ["step", "attack","loss", "accuracy", "f1-score", "mse", "tp", "fp", "tn", "fn", "precision", "recall", "auc", "mcc"]

if not os.path.exists(file_results):
    with open(file_results, mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        

if not os.path.exists("time_test.csv"):
    with open("time_test.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "attack", "database", "eps", "time"])

for step in range(100):
    test_ds = load_images(
        base_path,
        seed=int(time.time()),
        label_mode="categorical",
        image_size=image_size,
        subset="validation",
        validation_split=0.9,
        batch_size=32
    )

    test_ds_aux = test_ds.unbatch()
    images = np.asarray(list(test_ds_aux.map(lambda x, y: x)))
    labels = np.asarray(list(test_ds_aux.map(lambda x, y: y)))

    model, preprocess = InceptionV3(), preprocess_input
    model_trained = load_model(os.path.join("weights","model_{}_{}_clean.hdf5".format(args.model_name, args.database)), custom_objects={"f1_score": f1_score, "mcc": mcc})
    star_time = time.time()
    dict_result = model_trained.evaluate(test_ds, verbose=2)
    results_time = time.time()  - star_time
    r = [step, args.attack] + dict_result
    
    with open(file_results, mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(r)
    
    with open("time_test.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow([args.model_name, args.attack, args.database, args.eps, str(results_time)])
    
    print("Results: {}".format(r))
