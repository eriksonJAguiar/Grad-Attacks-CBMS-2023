##Import
import os
import time
import csv
import argparse
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras import backend as K
from tensorflow.keras.metrics import MeanAbsoluteError, TruePositives, TrueNegatives, FalseNegatives, FalsePositives, Recall, Precision, AUC, CategoricalAccuracy
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import Callback 
import tensorflow_addons as tfa
import math

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow import device
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers
import tensorflow as tf
import sklearn.metrics as mc
from sklearn.preprocessing import label_binarize

##Setup
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.1, patience=3, verbose=0)
early = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

batch_size = 32
epoches = 30
lr = 1e-4

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--database', type=str, required=False, help="str name of database: chest_xray or OCT")
parser.add_argument('--database_csv', type=str, required=False, help="str name of database: chest_xray or OCT")
parser.add_argument('--model_name', type=str, required=True, help="CNN name such as: inceptionv3, vgg16, or nasnet")
# Parse the argument
args = parser.parse_args()

class batch_timer(Callback):
    def __init__(self, file_train, file_test):
        super(batch_timer, self).__init__()
        self.file_test = file_test
        self.file_train = file_train
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time_train = time.time()    
    
    def on_epoch_end(self, epoch, logs=None):
        stop_time_train = time.time()
        time_train = stop_time_train-self.start_time_train
        with open(file=self.file_train, mode="a", encoding="UTF8", newline="") as csvW:
            writer = csv.writer(csvW)
            writer.writerow([str(epoch), str(time_train)])
    
    def on_test_begin(self, logs=None):
        self.start_time_test = time.time()

    def on_test_end(self, logs=None):
        stop_time_test = time.time()
        time_train = stop_time_test-self.start_time_test
        with open(file=self.file_test, mode="a", encoding="UTF8", newline="") as csvW:
            writer = csv.writer(csvW)
            writer.writerow([str(time_train)])

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

#Define metrics
def mcc(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def roc_auc(y_true, y_pred):
    
    y_lab_new = label_binarize(y_true, classes=list(range(4)))
    y_pred_adv_new = label_binarize(y_pred, classes=list(range(4)))
            
    auc = mc.roc_auc_score(y_lab_new, y_pred_adv_new, average='macro',multi_class='ovo') 
    
    return auc


METRICS = [
      #"accuracy",
      tfa.metrics.F1Score(name="f1_score", num_classes=4, average="macro"),
      CategoricalAccuracy(name="accuracy"),
      #f1_score,
      MeanAbsoluteError(name='mse'),
      #tfa.metrics.CohenKappa(num_classes = 4),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc', curve='ROC'),
      #roc_auc,
      mcc
]


class Model_Custom():
    
    def __init__(self) -> None:
        pass
    
    def get_model(self, model_name, class_num):
        
        model = None
        if model_name == "vgg16":
            model = self.vgg16(class_num)
        elif model_name == "inceptionv3":
            model = self.inceptionv3(class_num)
        elif model_name == "resnet50":
            model = self.resnet50(class_num)
        elif model_name == "xception":
            model = self.inceptionv3(class_num)
        
        return model
        
    
    def vgg16(self, class_num):
        
        base_model = VGG16(weights="imagenet", include_top=False)
       
        x = None
        
        db = args.database_csv.split(".")[0] if not args.database_csv is None else args.database
        db = db.split("_")[0]
        
        if db == "chest":
            for layer in base_model.layers:
                layer.trainable = False
            #base_model.trainale = False
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            #x = BatchNormalization()(x)

            x = Dense(1000, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
        
            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)
            
            opt = Adam(learning_rate = lr)
        
        elif db == "OCT":
            
            x = base_model.output
            #tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(x)
            x = Dropout(0.7)(x)
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)

            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)
            
            opt = Adam(learning_rate = lr)
        
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)
        
        return model
    
    def inceptionv3(self, class_num):
        
        base_model = InceptionV3(weights="imagenet", include_top=False)
        
        db = args.database_csv.split(".")[0] if not args.database_csv is None else args.database
        db = db.split("_")[0]
        
        if db == "chest":
            base_model.treinable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)

            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)

        
        if db == "OCT":
            base_model.treinable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)

            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)


        x = Dense(class_num, activation='softmax')(x)

        model = Model(inputs = base_model.input, outputs = x)

        opt = Adam(learning_rate = lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)
        
        return model
    
    def resnet50(self, class_num):
        
        base_model = ResNet50V2(weights="imagenet", include_top=False)
        
        db = args.database_csv.split(".")[0] if not args.database_csv is None else args.database
        db = db.split("_")[0]
        
        if db == "chest":
            for layer in base_model.layers:
                layer.trainable = False
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            x = Dense(1000, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)

            opt = Adam(learning_rate = lr)
        
        
        elif db == "OCT":
            
            for layer in base_model.layers:
                layer.trainable =  True    
        
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            
            x = Dense(1000, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)
            

            opt = Adam(learning_rate=lr)
       
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)
        
        return model

     
    def xception(self, class_num):
        
        base_model = Xception(weights="imagenet", include_top=False)
        
        db = args.database_csv.split(".")[0] if not args.database_csv is None else args.database
        db = db.split("_")[0]
        
        if db == "chest":
            for layer in base_model.layers:
                layer.trainable = False
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(64, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)

            opt = Adam(learning_rate = lr)
        
        
        elif db == "OCT":
            
            for layer in base_model.layers:
                layer.trainable =  True    
        
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            
            x = Dense(512, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(54, activation='relu')(x)
            #x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(class_num, activation='softmax')(x)

            model = Model(inputs = base_model.input, outputs = x)

            opt = Adam(learning_rate=lr)
       
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)
        
        return model
    

#dataset = ["chest_xray", "OCT"]
data = args.database if not args.database is None else args.database_csv.split(".")[0]

#for data in dataset:
image_train_folder = None
image_test_folder = None
train_csv = None
class_names = None

if not args.database is None:
    image_train_folder = os.path.join(".", data, "train_balanced")
    image_test_folder = os.path.join(".", data, "test_balanced")
    class_names = os.listdir(image_train_folder)
else:
    train_csv = pd.read_csv(args.database_csv)
    #df = train_csv.drop(columns=["x", "y", "Split"])
    df = train_csv.drop(columns=["x", "y"])
    class_names = list(df.keys())

name = args.model_name


print("Dataset {}".format(data))
print("Model trained {}".format(name))
image_size = (299, 299) if name == "inceptionv3" else (224, 224)
#image_size = (299, 299) if name == "inceptionv3" else (150, 150)
print("Imagew size {}".format(image_size))

class_num = len(class_names)
print("Class: {}".format(class_names))

models = {
    "inceptionv3": Model_Custom().get_model("inceptionv3", class_num), #input (299,299)
    "vgg16": Model_Custom().get_model("vgg16", class_num), #input (224,224)
    "resnet50": Model_Custom().get_model("resnet50", class_num), #input (224,224),
    "xception": Model_Custom().get_model("xception", class_num), #input (224,224),
}

model = models[name]

datagen_train = ImageDataGenerator(rescale = 1./255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True
                                )

datagen_test = ImageDataGenerator(rescale = 1./255)

##Build model
checkpoint = ModelCheckpoint("model_{}_{}_clean.hdf5".format(name, data), monitor='val_accuracy', verbose=0, save_best_only=True, mode='min')
logger = CSVLogger("results_{}_{}_clean.csv".format(name, data), separator=',', append=True)
        

if not args.database_csv is None:
        
    train_generator = datagen_train.flow_from_dataframe(dataframe = train_csv,
                                                    #directory = os.path.join("OCT", "train"),
                                                    x_col = "x",
                                                    y_col = class_names, # "y",
                                                    batch_size = batch_size,
                                                    #shuffle = True,
                                                    #class_mode = "raw",
                                                    class_mode="categorical",
                                                    target_size = image_size,
                                                )
       
    test_generator = datagen_train.flow_from_dataframe(dataframe = train_csv, 
                                                 #directory = os.path.join("OCT", "test"),
                                                 target_size = image_size,
                                                 shuffle = False,
                                                 x_col = "x",
                                                 subset="validation",
                                                 #y_col = class_names,
                                                 #class_mode = "raw",
                                                 y_col = class_names, #"y",
                                                 class_mode = "raw",
                                                 batch_size = batch_size
                                                ) 
    
    with device('/device:GPU:0'):
            history_net = model.fit(train_generator,
                            batch_size = batch_size,
                            epochs=epoches,
                            steps_per_epoch =  train_generator.n//batch_size,
                            validation_data=test_generator,
                            callbacks=[checkpoint, lr_reduce, logger, early, batch_timer("time_train_{}_{}.csv".format(data, name), "time_test_{}_{}.csv".format(data, name))]
                        )
else:
    train_generator = datagen_train.flow_from_directory(
            image_train_folder,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            seed=43,
            #subset="training"
    )

    test_generator = datagen_train.flow_from_directory(
            image_test_folder,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            #subset = "validation",
            seed=43
    )
    with device('/device:GPU:0'):
            history_net = model.fit(train_generator,
                            batch_size = batch_size,
                            epochs=epoches,
                            steps_per_epoch =  train_generator.n//batch_size,
                            validation_data=test_generator,
                            callbacks=[checkpoint, lr_reduce, logger, early, batch_timer("time_train_{}_{}.csv".format(data, name),  "time_test_{}_{}.csv".format(data, name))]
                        )

