import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import numpy as np

from random import shuffle
from PIL import Image

from pycocotools.coco import COCO

# set path for data (COCO expects it to be in "images")
dataDir="pcb-vision/data/"

# set path for full annotations 
annFile="pcb-vision/data/result.json"

# Initialize the COCO api for instance annotations
coco=COCO(annFile)

# santity check for categories
category_ids = coco.getCatIds()
num_categories = len(category_ids)
print('number of categories: ',num_categories)
for ids in category_ids:
    cats = coco.loadCats(ids=ids)
    print(cats)

# santity check for image info kept in annotation file
image_ids = coco.getImgIds()
print(len(image_ids))
image_id = image_ids[5]
image_info = coco.loadImgs(image_id)
print(image_info)

# Load annotations for the given ids
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)

#print(len(annotation_ids))

# make sure names match classes
filterClasses = ['bad_Cap', 'bad_IC', 'bad_Res', 'good_Cap', 'good_IC', 'good_Res']
catIds = coco.getCatIds(catNms=filterClasses)
print(catIds)

# santiy checks
catID = 2
print(coco.loadCats(ids=catID))

imgId = coco.getImgIds(catIds=[catID])[4]
print(imgId)

ann_ids = coco.getAnnIds(imgIds=[imgId], iscrowd=None)
print(ann_ids)

# santity check --> print out example annotation
print(f"Annotations for Image ID {imgId}:")
anns = coco.loadAnns(ann_ids)

image_path = dataDir + coco.loadImgs(imgId)[0]['file_name']
print(image_path)
image = plt.imread(image_path)
plt.imshow(image)

coco.showAnns(anns, draw_bbox=True)

plt.axis('off')
plt.title('Annotations for Image ID: {}'.format(image_id))
plt.tight_layout()
plt.show()

# main function that shows category IDs and info for a sample - check
def main():

    cat_ids = coco.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    query_id = cat_ids[0]
    query_annotation = coco.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    print("Category ID -> Category Name:")
    print(f"Category ID: {query_id}, Category Name: {query_name}")

    query_name = cat_names[2]
    query_id = coco.getCatIds(catNms=[query_name])[0]
    print("Category Name -> ID:")
    print(f"Category Name: {query_name}, Category ID: {query_id}")

    img_ids = coco.getImgIds(catIds=[query_id])
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    img_id = img_ids[2]
    img_info = coco.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    print(f"Image ID: {img_id}, File Name: {img_file_name}")

    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    print(f"Annotations for Image ID {img_id}:")
    print(anns)

    im = plt.imread(dataDir + coco.loadImgs(img_id)[0]['file_name'])
    plt.axis("off")
    plt.imshow(np.asarray(im))
    plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)
    plt.show()
    return

if __name__ == "__main__":

    main()

# check showing Category Distribution in Images --> corresponding figure - cat_image.png 
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

category_names = [cat['name'].title() for cat in cats]

category_counts = [coco.getImgIds(catIds=[cat['id']]) for cat in cats]
category_counts = [len(img_ids) for img_ids in category_counts]


colors = sns.color_palette('viridis', len(category_names))

plt.figure(figsize=(5, 5))
sns.barplot(x=category_counts, y=category_names, palette=colors)

for i, count in enumerate(category_counts):
    plt.text(count + 20, i, str(count), va='center')
plt.xlabel('Count',fontsize=20)
plt.ylabel('Category',fontsize=20)
plt.title('Category Distribution in Images',fontsize=25)
plt.tight_layout()
plt.savefig('coco-cats.png',dpi=300)
plt.show()

print(category_counts) #number of IMAGES containing these cats

# same info as above (Category Distribution in Images), just in pie chart form.
total_count = sum(category_counts)
category_percentages = [(count / total_count) * 100 for count in category_counts]


plt.figure(figsize=(4, 4))


labels = [f"{name} " for name, percentage in zip(category_names, category_percentages)]
label_props = {"fontsize": 25, 
               "bbox": {"edgecolor": "white", 
                        "facecolor": "white", 
                        "alpha": 0.7, 
                        "pad": 0.5}
              }

wedges, _, autotexts = plt.pie(category_counts, 
                              autopct='', 
                              startangle=90, 
                              textprops=label_props, 
                              pctdistance=0.85)

legend_labels = [f"{label}\n{category_percentages[i]:.1f}%" for i, label in enumerate(labels)]
plt.legend(wedges, legend_labels, title="Categories", loc="upper center", bbox_to_anchor=(0.5, -0.01), 
           ncol=4, fontsize=12)

plt.axis('equal')
plt.title('Category Distribution in COCO Dataset', fontsize=29)
plt.tight_layout()
plt.savefig('coco-dis.png', dpi=300)
plt.show()

import sklearn
import funcy
import argparse

# this part of the script uses the cocosplit.py file to break annotations into train and test
!python cocosplit.py -s 0.7 result.json train.json test.json

# now read the files in and get info for building the model
ANNOTATION_FILE_TRAIN = 'train.json'
ANNOTATION_FILE_VAL = 'test.json'

coco_train = COCO(ANNOTATION_FILE_TRAIN)
catIds_train = coco_train.getCatIds()
imgIds_train = coco_train.getImgIds()
imgDict_train = coco_train.loadImgs(imgIds_train)

coco_val = COCO(ANNOTATION_FILE_VAL)
catIds_val = coco_val.getCatIds()
imgIds_val = coco_val.getImgIds()
imgDict_val = coco_val.loadImgs(imgIds_val)

# check the image length - it should be 125 train/54 test with 6 classes
print(len(imgIds_train), len(catIds_train))
print(len(imgIds_val), len(catIds_val))
print(catIds_val)
print(coco_val.getImgIds())
print(coco_train.getImgIds())

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

import torch

# set the model parameters and make it cohesive for object detection
NUM_CLASSES = 6

def get_faster_rcnn_model(num_classes):
    """return model and preprocessing transform"""
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.cls_score.in_features,
        out_features=num_classes,
        bias=True,
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.bbox_pred.in_features,
        out_features=num_classes * 4,
        bias=True,
    )
    preprocess = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.transforms()
    return model, preprocess

model, preprocess = get_faster_rcnn_model(num_classes=NUM_CLASSES)

# check out the transformed model
print(model.transform)

# make PyTorch and COCO annotations compatible
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    # adapted from https://github.com/pytorch/vision/issues/2720

    def __init__(self, root, annFile, transform=None):
        """Load COCO annotation data."""
        self.data_dir = Path(root)
        self.transform = transform

        # load the COCO annotations json
        anno_file_path = annFile
        with open(str(anno_file_path)) as file_obj:
            self.coco_data = json.load(file_obj)
        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data["annotations"]:
            image_id = anno["image_id"]
            self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data["images"][index]
        image_id = image_data["id"]
        image_path = self.data_dir / image_data["file_name"]
        image = Image.open(image_path).convert("RGB")

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": [],
        }
        for anno in annos:
            coco_bbox = anno["bbox"]
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data["boxes"].append([left, top, right, bottom])
            anno_data["labels"].append(anno["category_id"])
            anno_data["area"].append(area)
            anno_data["iscrowd"].append(anno["iscrowd"])

        target = {
            "boxes": torch.as_tensor(anno_data["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(anno_data["labels"], dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(anno_data["area"], dtype=torch.float32),
            "iscrowd": torch.as_tensor(anno_data["iscrowd"], dtype=torch.int64),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
# make sure PyTorch can see the train/test datasets - should be 125 train / 54 test
import random
import torchvision.transforms as T
from IPython.display import display
from PIL import ImageDraw

# create datasets
training_dataset = CocoDataset(
    root="./",
    annFile="train.json",
    transform=preprocess,
)
validation_dataset = CocoDataset(
    root="./",
    annFile="test.json",
    transform=preprocess,
)

print(f"training dataset size: {training_dataset.__len__()}")

print(f"validation dataset size: {validation_dataset.__len__()}")

# check sample with bounding box - no label
img, label = training_dataset[random.randint(0, len(training_dataset) - 1)]
print(f"random training label: {label}")

transform = T.ToPILImage()
img = transform(img)
x1, y1, x2, y2 = label["boxes"].numpy()[0]
draw = ImageDraw.Draw(img)
draw.rectangle([x1, y1, x2, y2], fill=None, outline="#ff0000cc", width=2)
display(img)

# set batch size and return tuple data
BATCH_SIZE = 1

def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))

train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate,
)

params = [p for p in model.parameters() if p.requires_grad]

# build the optimizer
optimizer = torch.optim.SGD(
    params, 
    lr=0.001, 
    momentum=0.9, 
    weight_decay=0.0005
)

# train the model. takes around 40 minutes on Apple M2

num_epochs = 20
train_loss_list = []
validation_loss_list = []
model.train()
for epoch in range(num_epochs):
    N = len(train_loader.dataset)
    current_train_loss = 0
    # train loop
    for images, targets in train_loader:
#        images = list(image.to(device) for image in images)
#        targets = [
#            {
#                k: v.to(device) if isinstance(v, torch.Tensor) else v
#                for k, v in t.items()
#            }
#            for t in targets
#        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        current_train_loss += losses
    train_loss_list.append(current_train_loss / N)

    # validation loop
    N = len(validation_loader.dataset)
    current_validation_loss = 0
    with torch.no_grad():
#        for images, targets in validation_loader:
#            images = list(image.to(device) for image in images)
#            targets = [
#                {
#                    k: v.to(device) if isinstance(v, torch.Tensor) else v
#                    for k, v in t.items()
#                }
#                for t in targets
#            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            current_validation_loss += losses
    validation_loss_list.append(current_validation_loss / N)

    print(f"epoch: {epoch}")
    print(
        f"train loss: {train_loss_list[-1]}, validation loss: {validation_loss_list[-1]}"
    )

    # save the model as .pth. Plot the train loss val function for this model.
    torch.save(model, "./model.pth") # save model to file

# plot losses
train_loss = [x.cpu().detach().numpy() for x in train_loss_list]
validation_loss = [x.cpu().detach().numpy() for x in validation_loss_list]

plt.plot(train_loss, "-o", label="train loss")
plt.plot(validation_loss, "-o", label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

# load the model and make predictions
model = torch.load("./model.pth")

def inference(img, model):
    model.eval()
    pred = model([img]) # forward pass

    transform = T.ToPILImage()
    img = transform(img)
    x1, y1, x2, y2 = pred[0]["boxes"].cpu().detach().numpy()[0]
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=None, outline="#ff0000cc", width=2)
    display(img)
    return pred

img, _ = validation_dataset[random.randint(0, len(validation_dataset) - 1)]

inference(img, model)

# show the distribution of component annotations
import matplotlib.pyplot as plt

categories = ['good_Cap', 'good_Res', 'good_IC', 'bad_Cap', 'bad_Res', 'bad_IC']
values = [447, 369, 114, 90, 74, 30]

plt.figure(figsize=(8, 6))
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Component Annotations for PCB Dataset')
plt.axis('equal') 
plt.show()

# show the count for number of layers and top level layers
# Count the total number of layers (modules)
num_layers_all = sum(1 for _ in model.modules())
print(f'Total number of layers in FasterRCNN: {num_layers_all}')

# Count the number of top-level layers (e.g., sequential blocks)
num_layers_top = sum(1 for _ in model.children())
print(f'Number of top-level layers: {num_layers_top}')

