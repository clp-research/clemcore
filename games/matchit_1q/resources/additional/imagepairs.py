import argparse

import json
import pandas as pd
import random
from tqdm import tqdm
from SetSimilaritySearch import all_pairs

import re
import requests

import pathlib

import os
import glob
from torch.utils.data import Dataset
import clip
import torch
from PIL import Image


def filename_from_url(url):
    pattern = r'\d+.jpg'
    return re.findall(pattern,url)[-1]

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_file_names = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.transform = transform
        #self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image_file_name = self.image_file_names[idx]
        image = Image.open(image_file_name)
        if self.transform:
            image = self.transform(image)
        return image

def from_image_to_vector(x, process_fn):
    with torch.no_grad():
        image_features = model.encode_image(process_fn(x).unsqueeze(0))    
    return image_features

def cosine_function(x,y):
    return torch.nn.functional.cosine_similarity(embedding_dict[x], embedding_dict[y]).item()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", "-k", help="Process k images from Visual Genome.", type=int)
    parser.add_argument("--nlargest","-n",help = "Take n pairs with top jaccard-indices.",  type=int)
    args = parser.parse_args()

    n = args.nlargest

    dir_name = 'pictures'+ str(args.sample) + "_" + str(n)
    pathlib.Path(dir_name).mkdir(exist_ok=True) 

    os.environ["IMG_DIR"] = dir_name

    with open("attributes.json") as user_file:
            attsv120 = json.load(user_file)
    with open("image_data.json") as user_file:
            meta = json.load(user_file)

    url_lookup = {}
    for m in meta:
        url_lookup[m["image_id"]] = m["url"]

    print("Everything loaded.")
    # random sample of k pictures
    random.seed(0)
    sample = random.sample(range(len(attsv120)), k = args.sample)

    # making a dict with the k: image_id and v: contents of each image (object sysnets, object names and if applicable: attributes)
    img_conts_long = {}
    num_objs = []
    num_objs_unique = []
    for img in tqdm(sample, desc ="Filling image_conts_long dict"):
        contentlist = []
        for obj in attsv120[img]["attributes"]:
            contentlist += obj["synsets"]
            #contentlist += obj["names"]
            if "attributes" in obj:
                contentlist += obj["attributes"]
            img_conts_long[attsv120[img]["image_id"]]= contentlist               # !!! image_ids do not necessarily correspond to the index in the attsv120 dict !!!
        # how many words per picture?
        num_objs.append(len(contentlist))
        # how many unique words per picture?
        num_objs_unique.append(len(set(contentlist)))

    # uncomment, if image annotations should be saved
    # with open("img_conts_long.json", "w") as f:
    #     json.dump(img_conts_long , f) 

    # remove entries, whith <25 and >75 unique description words -> have a minimum amount of description words for comparison and it is better to have smaller sets for scalability https://github.com/ekzhu/SetSimilaritySearch/issues/11 (although our set sizes are not that extreme)
    img_conts = {k:v for (k,v) in img_conts_long.items()if len(set(v))>25 and len(set(v))<75}
   
    # uncomment, if image annotations should be saved
    # with open("img_conts.json", "w") as f:
    #     json.dump(img_conts, f) 


    contlist = list(img_conts.items())
    newlist = [set(y) for (_,y) in contlist]
    print("calculating jaccard similarities")
    setsimlist = list(all_pairs(newlist, similarity_func_name="jaccard", 
            similarity_threshold= 0))

    print("Jaccards indices are calculated.")


    big_df = pd.DataFrame(setsimlist, columns = ["pic1", "pic2", "jac_ind"])
    big_df["id1"] = [contlist[i][0] for i in big_df.pic1]
    big_df["id2"] = [contlist[i][0] for i in big_df.pic2]
    big_df["jac_ind"] = big_df["jac_ind"].round(4)
  

    big_df.to_csv(str(args.sample)+"_pics_sample_jaccards.zip", columns = ["id1", "id2", "jac_ind"], index = False, encoding = "utf-8")
    
    #adding urls later because I don't need to save them
    big_df["url1"] = [url_lookup[i] for i in big_df.id1]
    big_df["url2"] = [url_lookup[i] for i in big_df.id2]

    #dropping duplicates such that every picture appears at most twice, favoring high jaccard indices
    big_df = big_df.sort_values("jac_ind").drop_duplicates(subset=["id2"], keep= "last").drop_duplicates(subset = ["id1"], keep = "last")
    big_df2 = big_df.sort_values("jac_ind").drop_duplicates(subset=["id1"], keep= "last").drop_duplicates(subset = ["id2"], keep = "last")

    #nlargest of longer df -> ! Removed the jac_ind above 0.22 condition in favor of small samples (for large samples and rather small n, this should not matter)
    if len(big_df) > len(big_df2):
        n_largest = big_df.nlargest(n, "jac_ind")
    else:
        n_largest = big_df2.nlargest(n, "jac_ind")
    n_largest.to_csv(str(n)+"_largest.csv")


    urls = set(n_largest.url1).union(set(n_largest.url2))
    # downloading n pictures into pictures folder -> make sure that empty before, or check whether this would be a problem
    for url in tqdm(urls, desc ="downloading pictures"):
        img_data = requests.get(url).content
        with open(dir_name + '/' + filename_from_url(url), 'wb') as handler:
            handler.write(img_data)


    ######################## CLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image_dataset = ImageDataset(img_dir=os.environ["IMG_DIR"], transform=preprocess)
    # next(iter(image_dataset)).shape

    print("making image dataset")

    image_dataset = ImageDataset(
        img_dir=os.environ["IMG_DIR"], 
        transform=lambda x: from_image_to_vector(x, process_fn=preprocess)
    )

    embedding_dict = {}

    picturenames = [f for f in os.listdir(os.environ["IMG_DIR"]) if not f.startswith('.')]          ### hier wäre es echt gut, wenn vor dem Bilderdownload sichergegangen wird, dass in dem Ordner sonst goanix ist!

    print("Making embedding dict")
    for i in range(len(picturenames)):
        embedding_dict[int(picturenames[i].split(sep=".")[0])]=image_dataset.__getitem__(i)

    print("Calculating clip score.")

    n_largest['clipscore'] = n_largest.apply(lambda x: cosine_function(x.id1, x.id2), axis=1)

    n_largest.to_csv(str(n)+"_largest_clipscore.csv")

    print("Done.")