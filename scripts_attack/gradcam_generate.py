import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from tensorflow.keras.applications.inception_v3 import preprocess_input as ip
from tensorflow.keras.applications.vgg16 import preprocess_input as vp
from keras import backend as K
import argparse
import os
import pathlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--attack_db', type=str, required=True)
parser.add_argument('--original', type=bool, required=False, action="store_true")

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


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam_processing(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    #tf.keras.preprocessing.image.save_img(cam_path, superimposed_img)
    return superimposed_img
    #superimposed_img.save(cam_path)
    #plt.savefig(superimposed_img, cam_path, bbox_inches='tight')
    #return superimposed_img
    
    # Save the superimposed image
    #superimposed_img.save(cam_path)
    

def get_grad_cam(img_path, model, model_name, db, last_conv_layer_name, image_size, to_path):
    cl = ['DRUSEN', 'CNV', 'DME', 'NORMAL'] if db == "OCT" else ["NORMAL", "PNEUMONIA"]
    
    model.layers[-1].activation = None 
    img_path = str(img_path)
   
    if args.original:
        orig_path = os.path.join(db, "test_balanced", str(img_path).split("/")[-2], str(img_path).split("/")[-1])
        img_array = rp(get_img_array(orig_path, size=image_size)) if model_name == "resnet50" else ip(get_img_array(orig_path, size=image_size)) if model_name == "inceptionv3" else vp(get_img_array(orig_path, size=image_size))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        grad  = gradcam_processing(orig_path, heatmap)
    
    else:
        img_array = rp(get_img_array(img_path, size=image_size)) if model_name == "resnet50" else ip(get_img_array(img_path, size=image_size)) if model_name == "inceptionv3" else vp(get_img_array(img_path, size=image_size))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        grad  = gradcam_processing(img_path, heatmap)

    
    if not os.path.exists(os.path.join(to_path, str(img_path).split("/")[-2])):
        os.makedirs(os.path.join(to_path, str(img_path).split("/")[-2]))

    tf.keras.preprocessing.image.save_img(os.path.join(to_path, str(img_path).split("/")[-2], str(img_path).split("/")[-1]), grad)
    #tf.keras.preprocessing.image.save_img(os.path.join(to_path, str(img_path).split("/")[-2], "heatmap_{}".format(str(img_path).split("/")[-1])), heatmap)
    #cv2.imwrite(os.path.join(to_path, str(img_path).split("/")[-2], "heatmap_{}".format(str(img_path).split("/")[-1])), heatmap)
    plt.matshow(heatmap)
    plt.axis('off')
    plt.savefig(os.path.join(to_path, str(img_path).split("/")[-2], "heatmap_{}".format(str(img_path).split("/")[-1])), bbox_inches='tight', pad_inches=0)
    

#imgs_path_root = "attacks_images"
imgs_path_root = args.attack_db
image_size = (224, 224)
#files = glob.glob(os.path.join(glob.escape(p) + "/*.jpeg"), recursive=True)
#img = os.path.join("attacks_images", "Deep_resnet50_chest_xray_attack", "0.1", "PNEUMONIA", "person51_virus_105.jpeg")
#last_conv_layer_name = "conv5_block3_out" # Block Resnet50
last_conv_layer_name = {
    "resnet50": "conv5_block3_out", 
    "inceptionv3": "conv2d_93",
    "vgg16": "block5_conv3"
}

epsilons = [0.01] if args.original else [0.1, 0.01, 0.05, 0.075]

for db in ["chest_xray", "OCT"]:
    for cnn in ["resnet50", "inceptionv3", "vgg16"]:
        model_trained = load_model(os.path.join("weights","model_{}_{}.hdf5".format(cnn, "chest" if db == "chest_xray" else db)), custom_objects={"f1_score": f1_score, "mcc": mcc})
        for attack in ["FGSM", "PGD", "Deep"]:    
            for eps in epsilons:
                print("Running for {} -- {} -- {}".format(db, attack, str(eps)))
                p_from = os.path.join(imgs_path_root, "{}_{}_{}_attack".format(attack, cnn, db), str(eps))
                
                files = list(pathlib.Path(p_from).rglob("*.jpeg"))
                if args.original:
                    p_to = os.path.join("clean_grad", cnn, db)
                else:
                    p_to = os.path.join("gradcam", "{}_{}_{}_attack".format(attack, cnn, db), str(eps))
                
                if not os.path.exists(p_to):
                    os.makedirs(p_to)
                
                [get_grad_cam(f, model_trained, cnn, db, last_conv_layer_name[cnn], image_size, p_to) for f in files]
            
            
            