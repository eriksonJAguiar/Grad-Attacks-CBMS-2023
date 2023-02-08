import glob
import shutil
import os
import argparse


parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--database', type=str, required=True)
parser.add_argument('--attack_path', type=str, required=True)
parser.add_argument('--eps', type=str, required=True)
# Parse the argument
args = parser.parse_args()

attack_path = os.path.join(args.attack_path, args.eps)
print("Path of images generated by attack : {}".format(attack_path))

from_path = os.path.join("{}".format(args.database), "test")
print("Orignal test path : {}".format(from_path))

to_path = os.path.join(args.database, args.attack_path, args.eps)
print("Attack test path : {}".format(to_path))
#to_path = os.path.join("original_xray", "FGSM")

if not os.path.exists(os.path.join(args.database, args.attack_path)):
    os.mkdir(os.path.join(args.database, args.attack_path))

if not os.path.exists(os.path.join(args.database, args.attack_path, args.eps)):
    os.mkdir(os.path.join(args.database, args.attack_path, args.eps))

cl = os.listdir(from_path)
print("class {}".format(cl))
os.mkdir(os.path.join(to_path, cl[0]))
os.mkdir(os.path.join(to_path, cl[1]))

attack_images = glob.glob(os.path.join(attack_path, "*","*.jpeg"), recursive=True)
images_original = glob.glob(os.path.join(from_path, "*", "*.jpeg"), recursive=True)

for p in images_original:
    im = p.split("/")[-1]
    c = p.split("/")[-2]
    shutil.copy2(os.path.join(from_path, c, im), os.path.join(to_path, c, im))