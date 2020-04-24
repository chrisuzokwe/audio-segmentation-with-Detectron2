# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import colabtools

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# more common libraries for retrieving data
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import audioread
import json
import os
import itertools
from PIL import Image
import cv2

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
""" compute the overlap accuracy of the diagonal image labels
 return [# of labels above threshold] & [average overlap] """
def diagonal_acc(dataset_dict, outputs, threshold):
    totalacc = 0
    numlabels = 0

    # find predicted bbox coordinates
    dpred = []
    for box in outputs["instances"].pred_boxes[0].tensor[0]:


        print(box)


        tmp = []



        if abs(box.item() - box.item()) < 5:
            for x in box:
                tmp.append(x.item())
            dpred.append(tmp)

    ##get diagonal dataset outline values (where x1 = y1)
    for prediction in dpred:
        acc = 0
        diff = 100
        for gtruth in dataset_dict["annotations"]:
            if gtruth["bbox"][0] == gtruth["bbox"][1]: # if diagonal entry, calculate smallest similarity

                for i in range(4):
                    tmp = abs(gtruth["bbox"][i] - prediction[i])

                if tmp < diff:
                    nearest_seg = gtruth["bbox"]
                    diff = tmp
        # calculate accuracy from nearest segment
        if prediction[2] < nearest_seg[0] or prediction[3] < nearest_seg[0]:
            acc = 0
        elif prediction[0] > nearest_seg[2] or prediction[1] < nearest_seg[2]:
            acc = 0
        else:
            overlaparea = (max(nearest_seg[0], prediction[0]) - min(nearest_seg[2],prediction[2])) * (max(nearest_seg[1], prediction[1]) - min(nearest_seg[3],prediction[3]))
            acc = (prediction[2]-prediction[0])*(prediction[3]-prediction[1])/overlaparea

        if acc > threshold:
            numlabels = numlabels + 1
            totalacc = totalacc + acc

    return numlabels, totalacc/numlabels








""" # compute the f1 score using diagonal(time) label entries. return precision, recall, and f1
def diagonal_f1(dataset_dict, outputs):
    ##find predicted bbox coordinates

    ##get diagonal dataset outline values (where x1 = y1)



def segment_diagonal_acc(dataset_dict, outputs): """