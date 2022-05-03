import os
import json
import torch
import numpy as np
import argparse
import time
import requests
import torchvision

from PIL import Image
from torchvision import transforms as T
from torchvision.io import read_image
from surfree import SurFree


def get_model():
    model = torchvision.models.resnet18(pretrained=True).eval()
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    return torch.nn.Sequential(normalizer, model).eval()

def get_imagenet_labels():
    response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
    return eval(response.content)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results_test/", help="Output folder")
    parser.add_argument("--n_images", "-n", type=int, default=2, help="N images attacks")
    parser.add_argument(
        "--config_path", 
        default="config_example.json", 
        help="Configuration Path with all the parameter for SurFree. It have to be a dict with the keys init and run."
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ###############################
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        raise ValueError("{} doesn't exist.".format(output_folder))

    ###############################
    print("Load Model")
    model = get_model()

    ###############################
    print("Load Config")
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise ValueError("{} doesn't exist.".format(args.config_path))
        config = json.load(open(args.config_path, "r"))
    else:
        config = {"init": {}, "run": {"epsilons": None}}

    ###############################
    print("Get understandable ImageNet labels")
    imagenet_labels = get_imagenet_labels()
    
    ###############################
    print("Load Data")
    X = []
    transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
    for img in os.listdir("./images"):
        X.append(transform(read_image(os.path.join("./images", img))).unsqueeze(0))
    X = torch.cat(X, 0) / 255
    y = model(X).argmax(1)

    ###############################
    print("Attack !")
    time_start = time.time()

    f_attack = SurFree(**config["init"])

    if torch.cuda.is_available():
        model = model.cuda(0)
        X = X.cuda(0)
        y = y.cuda(0)

    advs = f_attack(model, X, y, **config["run"])
    print("{:.2f} s to run".format(time.time() - time_start))

    ###############################
    print("Results")
    labels_advs = model(advs).argmax(1)
    nqueries = f_attack.get_nqueries()
    advs_l2 = (X - advs).norm(dim=[1, 2]).norm(dim=1)
    for image_i in range(len(X)):
        print("Adversarial Image {}:".format(image_i))
        label_o = int(y[image_i])
        label_adv = int(labels_advs[image_i])
        print("\t- Original label: {}".format(imagenet_labels[str(label_o)]))
        print("\t- Adversarial label: {}".format(imagenet_labels[str(label_adv)]))
        print("\t- l2 = {}".format(advs_l2[image_i]))
        print("\t- {} queries\n".format(nqueries[image_i])) 

    ###############################
    print("Save Results")
    for image_i, o in enumerate(X):
        o = np.array(o * 255).astype(np.uint8)
        img_o = Image.fromarray(o.transpose(1, 2, 0), mode="RGB")
        img_o.save(os.path.join(output_folder, "{}_original.jpg".format(image_i)))

        adv_i = np.array(advs[image_i] * 255).astype(np.uint8)
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB")
        img_adv_i.save(os.path.join(output_folder, "{}_adversarial.jpg".format(image_i)))