import os
import random
import pickle
from glob import glob

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as tt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import PIL
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


config = DotDict()
config.image_size = 224
config.num_classes = 40
config.batch_size = 4
config.epochs = 100
config.lr = 1e-6
config.momentum = 0.9
config.train_p = 0.80

config.rotate = 15
config.fill = 248
config.perspective = 0.3
config.sharpness = 0.9
config.contrast = 0.3


DS_PATH = 'toadbox/'
# imagenet
STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


import torchvision.transforms.functional as ff
from PIL import Image
import random

def centerScale(min=.22, max=.35):
    def func(img):
        H, W = img.size
        scale = random.uniform(min, max)
        h = scale*H
        w = scale*W 
        t = (H - h) / 2
        l = (W - w) / 2

        img = ff.crop(img, t, l, h, w)
        return img
    return func

transform_test=tt.Compose([
                    #centerScale(.28, .28),
                    tt.Resize((config.image_size, config.image_size)),
                    #clahe(),
                    tt.ToTensor(),
])

transform_test=tt.Compose([
                    # tt.RandomRotation(config.rotate, fill=config.fill),
                    # tt.RandomPerspective(config.perspective, fill=config.fill),
                    tt.RandomAdjustSharpness(config.sharpness),
                    tt.RandomAutocontrast(config.contrast),
                    ## centerScale(),
                    tt.ColorJitter(0.4, 0.2, 0.1, 0.08),
                    ## tt.AutoAugment(tt.autoaugment.AutoAugmentPolicy.IMAGENET),
                    tt.Resize((config.image_size, config.image_size)),
                    ## clahe(),
                    tt.ToTensor(),
                    ## tt.Normalize(*STATS)
                 ])

class Siamese(nn.Module):
    def __init__(self, latent=32, mode='triplet', pretrained=True):
        super().__init__()
        self.mode = mode
        self.backbone = torchvision.models.resnet101(pretrained=pretrained)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(2048, latent)

    def forward_single(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def forward(self, left, right, neg=None):
        if self.mode == 'pair':
            left = self.forward_single(left)
            right = self.forward_single(right)
            return left, right
        elif self.mode == 'triplet':
            if neg is None:
                raise ValueError('Provide negative for triplet loss')
            a = self.forward_single(left)
            p = self.forward_single(right)
            n = self.forward_single(neg)
            return a, p, n

def bake_embeddings(model, path=DS_PATH, transform=transform_test):
    """
    Returns a dictionary of computed embeddings for the directory given
    """
    baked = {}

    model.eval()
    model.to(device)

    print('Baking embeddings!')
    count = len(list(os.walk(path)))
    for i, (dp, dn, fnames) in tqdm(enumerate(os.walk(path)), total=count):
        folder = dp.split('/')[-1]
        if folder == '':
            continue
        refs = []
        with torch.no_grad():
            names = list(map(lambda n: os.path.join(dp, n), fnames))
            for name in names:
                ref = transform(Image.open(name)).unsqueeze(0).to(device)
                ref = model.forward_single(ref).cpu()
                refs.append((name, ref))
        baked[folder] = refs

    model.cpu()
    return baked

def bake_distance(sample, baked):
    dists = {}
    for i, (folder, embeddings) in enumerate(baked.items()):
        dists[folder] = [(pair[0], torch.dist(sample, pair[1]).item()) for pair in embeddings]
    return dists

def siam_rank(model, baked_dist, sample, kind='median'):
    """
    Returns sorted list of tuples ranking predictions for sample
    compared to baked embeddings with different criterion.

    """
    model.eval()
    model.to(device)

    ranking = {}
    for folder, name_dist in baked_dist.items():
        dist = [p[1] for p in name_dist]
        if kind == 'mean':
            dist = np.mean(dist)
        elif kind == 'min':
            dist = min(dist)
        elif kind == 'max':
            dist = max(dist)
        elif kind == 'median':
            dist = np.median(dist)
        elif kind == 'min_fair':
            dist = sorted(dist)[1]
        else:
            raise ValueError()

        ranking[folder] = dist

    ranking = sorted(ranking.items(), key=lambda x: x[1])
    model.cpu()
    return ranking
        

def mine_hardneg(baked_dist, sample_path, kind='mean', k=5, hardness='semihard'):
    """
    Returns sorted list of hard/semi-hard negative for the sample given
    """
    positive_class = sample_path.split('/')[-2]

    negatives = []
    positives = []

    for name, pairs in baked_dist.items():
        if name == positive_class:
            positives += pairs
        else:
            negatives += pairs
    positives = [p[1] for p in positives]
    threshold = 0
    if kind == 'mean':
        threshold = np.mean(positives)
    elif kind == 'min':
        threshold = min(positives)
    elif kind == 'max':
        threshold = max(positives)
    elif kind == 'median':
        threshold = np.median(positives)
    elif kind == 'min_fair':
        threshold = sorted(positives)[1]
    else:
        raise ValueError()

    negatives = sorted(negatives, key=lambda x: x[1])

    if hardness == 'semihard':
        i = 0
        for n in negatives:
            if n[1] > threshold:
                break
            i += 1
        selected = negatives[i:i+k]
        if len(selected) == 0: selected = negatives[:k]
    elif hardness == 'hard':
        selected = negatives[:k]

    return selected

def infere_image(model, sample_pil, transform=transform_test):
    model.eval().to(device)
    sample = transform(sample_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        sample = model.forward_single(sample).cpu()
    model.cpu()
    return sample

def crop_save(ds_path, outdir='toadbox_crop'):
    for t in tqdm(os.listdir(ds_path)):
        batch_names = os.listdir(os.path.join(ds_path, t))
        crop_names = [os.path.join(outdir, t, x) for x in batch_names]
        for dir in [os.path.join(outdir, t) for x in batch_names]:
            os.makedirs(dir, exist_ok=True)
        batch_names = [os.path.join(ds_path, t, x) for x in batch_names]
        preds = yolo_model(batch_names)

        for name, crop_name, bboxes in zip(batch_names, crop_names, preds.xyxy):
            if len(bboxes) == 0:
                print(f'{name} empty!')
                continue
            best_pred = sorted(bboxes, key=lambda bbox: bbox[4], reverse=True)[0]
            bbox = best_pred[:4].tolist()
            crop = Image.open(name).crop((*bbox, ))
            crop.save(crop_name)

def crop_best(yolo_out):
    best_score = 0
    img = None
    for crop in yolo_out:
        score = crop['conf'].item()
        if score > best_score:
            best_score = score
            img = crop['im']
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = transform_test(img)
    return img

def yolo_process(sample):
    out = yolo_model(sample)
    img = crop_best(out)
    return img

model = Siamese(512, pretrained=False)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_model.pt')
model.load_state_dict(torch.load('weights.pth', map_location=torch.device('cpu')))

if not os.path.exists('baked.pkl'):
    baked = bake_embeddings(model, 'toadbox/')
    with open('baked.pkl', 'wb') as fp:
        pickle.dump(baked, fp)
else:
    print('USING EXISTING baked.pkl')

with open('baked.pkl', 'rb') as f:
    baked = pickle.load(f)

for p in model.parameters():
    p.requires_grad = False

def infere(t, model, topk=5):
    #img = yolo_process(t)
    img = t
    sample = infere_image(model, img)
    baked_dist = bake_distance(sample, baked)
    rank = siam_rank(model, baked_dist, sample)
    preds = [r[0] for r in rank[:topk]]
    images = [Image.open(os.path.join(DS_PATH, p, random.choice(os.listdir(f'{DS_PATH}/{p}')))) for p in preds]
    descriptions = ["Each toad can have it's own description"] * topk
    print(preds)
    return preds, images, descriptions


print('API READY!')
