"""
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python voc.py train --dataset=/home/ubuntu/Documents/datasets/VOCdevkit  --year=2007 --model=coco


    # Run COCO evaluatoin on the last model you trained
    python voc.py evaluate --dataset=/home/ubuntu/Documents/datasets/VOCdevkit  --year=2007 --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import xml.etree.ElementTree as ET
import skimage.draw
import re

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join('/home/ubuntu/Documents/datasets/models', "ResNet101-mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VocConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "voc"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################


class VocDataset(utils.Dataset):


    def load_voc(self, dataset_dir, subset, year, limit=100, return_coco=False):


        dataset_dir = os.path.join(dataset_dir, "VOC"+year)

        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                       'sofa', 'train', 'tvmonitor']

        for i, class_name in enumerate(class_names):
            self.add_class("voc", i+1, class_name)


        if subset == "train":
            with open(dataset_dir+"/ImageSets/Main/train.txt") as f:
                for line in f.readlines():
                    self.add_image("voc", image_id=line.strip(),
                                    path=os.path.join(dataset_dir, "JPEGImages/{}.jpg".format(line.strip())))
        elif subset == "val":
            xml_files = []
            with open(dataset_dir+"/ImageSets/Main/val.txt") as f:
                if return_coco:
                    for line in f.readlines():
                        xml_files.append(line.strip())
                else:
                    for line in f.readlines():
                        self.add_image("voc", image_id=line.strip(),
                            path=os.path.join(dataset_dir, "JPEGImages/{}.jpg".format(line.strip())))

        if return_coco:

            voc = COCO()
            voc.dataset["images"] = []
            voc.dataset["annotations"] = []
            voc.dataset["categories"] = [{"id": 1, "name": 'aeroplane'},{"id": 2, "name": 'bicycle'},{"id": 3, "name": 'bird'},{"id": 4, "name": 'boat'},{"id": 5, "name": 'bottle'},{"id": 6, "name": 'bus'},{"id": 7, "name": 'car'},{"id": 8, "name": 'cat'},{"id": 9, "name": 'chair'},{"id": 10, "name": 'cow'},{"id": 11, "name": 'diningtable'},{"id": 12, "name": 'dog'},{"id": 13, "name": 'horse'},{"id": 14, "name": 'motorbike'},{"id": 15, "name": 'person'},{"id": 16, "name": 'pottedplant'},{"id": 17, "name": 'sheep'},{"id": 18, "name": 'sofa'},{"id": 19, "name": 'train'},{"id": 20, "name": 'tvmonitor'}]

            for xml_file in xml_files[:limit]:
                tree = ET.parse(os.path.join(dataset_dir, "Annotations/{}.xml".format(xml_file)))

                image_size = tree.findall('size')[0]
                height = int(image_size.find('height').text)
                width = int(image_size.find('width').text)

                dataset_image_id = int("".join(re.findall("\d+", xml_file))[-7:])
                objs = tree.findall('object')
                for ix, obj in enumerate(objs):
                    bbox = obj.find('bndbox')
                    # Make pixel indexes 0-based
                    x1 = int(bbox.find('xmin').text) - 1
                    y1 = int(bbox.find('ymin').text) - 1
                    x2 = int(bbox.find('xmax').text) - 1
                    y2 = int(bbox.find('ymax').text) - 1

                    voc.dataset["annotations"].append({
                        "bbox": [x1+1, y1+1, x2-x1, y2-y1],
                        "image_id": dataset_image_id,
                        "category_id": class_names.index(obj.find("name").text) + 1,
                        "file_name": xml_file+'.jpg',
                        "id": int(str(x1)+str(y1)+str(x2)),
                        "area": (x2-x1)*(y2-y1),
                        "iscrowd": 0})

                voc.dataset["images"].append({
                            "file_name": xml_file+'.jpg', # "file_name": "COCO_val2014_000000382030.jpg",
                            "height": height,
                            "width": width,
                            "id": dataset_image_id})
                self.add_image(
                    "voc",
                    image_id=dataset_image_id,
                    path=os.path.join(dataset_dir, "JPEGImages/{}.jpg".format(xml_file)))

            voc.createIndex()
            return voc


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "voc":
            return super(self.__class__, self).load_mask(image_id)

        path_file = os.path.split(image_info["path"])
        tree = ET.parse(path_file[0][:-10]+"Annotations/"+path_file[1][:-3]+"xml")

        objs = tree.findall('object')
        image_size = tree.findall('size')[0]
        height = int(image_size.find('height').text)
        width = int(image_size.find('width').text)
        num_objs = len(objs)

        class_name = []
        mask = np.zeros([height, width, num_objs], dtype=np.uint8)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            class_name.append(self.class_names.index(obj.find('name').text.lower().strip()))
            rr, cc = skimage.draw.polygon([y1, y2, y2, y1], [x1, x1, x2, x2])
            mask[rr, cc, ix] = 1

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.array(class_name).astype(np.int8)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "voc":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  COCO Evaluation
############################################################


def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "voc"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": None
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = voc.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(voc, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        default="evaluate",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="/media/ubuntu/HDD1/dataset/VOCdevkit/",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default="2012",
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        default=COCO_MODEL_PATH,
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VocConfig()
    else:
        class InferenceConfig(VocConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.model.lower() == "voc":
        model_path = COCO_MODEL_PATH
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)
    else:
        model_path = args.model
        model.load_weights(model_path, by_name=True)
    # Load weights
    print("Loading weights ", model_path)



    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VocDataset()
        dataset_train.load_voc(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VocDataset()
        dataset_val.load_voc(args.dataset, "val", year=args.year)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = VocDataset()
        #val_type = "val" if args.year in '2017' else "minival"
        voc = dataset_val.load_voc(args.dataset, "val", year=args.year,
                            limit=int(args.limit), return_coco=True)

        dataset_val.prepare()
        print("Running VOC evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, voc, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))