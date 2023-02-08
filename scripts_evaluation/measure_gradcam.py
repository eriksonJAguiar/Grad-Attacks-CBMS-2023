import pandas as pd
import numpy as np
import argparse
import os
import csv
import tensorflow as tf
import pathlib

parser = argparse.ArgumentParser()
# Add an argument
#parser.add_argument('--original_csv', type=str, required=True)
parser.add_argument('--database_name', type=str, required=True)
#parser.add_argument('--path_attack', type=str, required=True)
#parser.add_argument('--attacks', type=str, required=True)
#parser.add_argument('--model', type=str, required=True)
#parser.add_argument('--attack_path', type=str, required=True)
#parser.add_argument('--eps', type=str, required=True)
# Parse the argument
args = parser.parse_args()

def preprocess(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    #image = slide.read_region(region, 0, size)
    # convert the image to np.array
    image = tf.image.resize(np.array(image), (224, 224))
    return image

#attacks = ["FGSM", "PGD", "Deep", "UAP"]
attacks = ["FGSM", "PGD", "Deep"]
models = ["vgg16", "inceptionv3", "resnet50"]
eps = [0.1, 0.01, 0.05, 0.075]
#db = "OCT"
db = args.database_name

#orignal_db = pd.read_csv(args.original_csv)

def get_images_path(path_attack, attack, model, database_name, eps):
    img_path_attack = os.path.join(path_attack, "{}_{}_{}_attack".format(attack, model, database_name), str(eps))
    #img_path_orignal =  os.path.join("database_name", "test_balanced")
    #img_atck = pd.DataFrame()
    #img_atck["x"] = glob.glob(os.path.join(img_path_attack, "*.jpeg"), recursive=True)
    #img_atck["y"] = [p.split("/")[-2] for p in img_atck["x"]]
    files = []
    for fn in pathlib.Path(img_path_attack).rglob(os.path.join("*", "*.jpeg")):
        fn = str(fn)
        cl = fn.split("/")[-2]
        heat = fn.split("/")[-1].split("_")[0]
        if not heat == "heatmap":
            files.append({"x":fn, "y": cl})
        
    #print(len(files))
    img_atck = pd.DataFrame.from_dict(files)
    
    #img_orig = pd.DataFrame()
    #img_orig["x"] = glob.glob(os.path.join(img_path_orignal, "*.jpeg"))
    #img_orig["y"] = [p.split("/")[-2] for p in img_orig["x"]]
    
    return img_atck
     
    
        
if not os.path.exists("grad_metrics.csv"):
    with open("grad_metrics.csv", mode="a") as f:
        w = csv.writer(f)
        w.writerow(["database", "model", "attack", "eps", "nissim", "MOD", "VID"])

for model in models:
    for attack in attacks:
        for e in eps:
            attack_db = get_images_path(os.path.join("attacks_images", "gradcam"), attack, model, db, e)
            nissim = list()
            for i in range(attack_db.shape[0]):
                atck_img = attack_db["x"].iloc[i]
                #print(atck_img)
                attacked_image = preprocess(atck_img)
                img_name = atck_img.split("/")[-1]
                cl = atck_img.split("/")[-2]
                orig_path = os.path.join("attacks_images", "clean_grad", model, db, cl, img_name)
                #attacked_image_path = os.path.join(args.path_attack, "{}_{}_{}_attack".format(attack, model, args.database_name), str(e), orignal_db["y"].iloc[i])
                #attacked_image = tf.keras.preprocessing.image.load_img(os.path.join(attacked_image_path, img))
                #if not os.path.exists(os.path.join(attacked_image_path, img)):
                #    continue
                #print(orig_path)
                #print("Attack : {}".format(atck_img))
                #print("Original: {}".format(orig_path))
                img_original = preprocess(orig_path)
                #im1 = tf.image.convert_image_dtype(img_original, tf.float32)
                #im2 = tf.image.convert_image_dtype(attacked_image, tf.float32)
                #psnr = float(tf.image.psnr(img_original, attacked_image, max_val=255))
                ssmi = float(tf.image.ssim(img_original, attacked_image, max_val=255))
                #im1 = tf.image.convert_image_dtype(img_original, tf.float32)
                #im2 = tf.image.convert_image_dtype(attacked_image, tf.float32)
                #psnr = tf.image.psnr(im1, im2, max_val=1.0)
                #ssmi = float(tf.image.ssim(im1, im2, max_val=1.0))
                
                nissim.append((1 - ssmi)/2)
            
            nissim = np.array(nissim)
            mod = np.mean(nissim)
            mh = (1/e)*np.sum(nissim)
            vid = np.sqrt((np.sum(nissim - mh)**2)/e) 
            
            print("Model: {} -- eps: {} -- attack: {}".format(model, e, attack))
            print("NISSMI: {} -- MOD: {} -- VID: {}".format(np.mean(np.array(nissim)), mod, vid))
            values = [db, model, attack, e, np.max(nissim), mod, vid]
                #print(values)
            with open("grad_metrics.csv", mode="a") as f:
                w = csv.writer(f)
                w.writerow(values)
