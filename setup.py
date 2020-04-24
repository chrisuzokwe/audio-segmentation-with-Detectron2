# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

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

# return 3 self similarity matrices
def get_3ssm(audio_path):
    # load audio into np array
    y, sr = librosa.load(audio_path)

    # using these characteristics to achieve ~5 fps
    hop_length = 8192
    n_fft = 16384

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # we can change the hop and fft size to get a matrix i dont need to change

    # compute chroma features and ssm
    chroma = librosa.feature.chroma_stft(S=S)
    ssm_chroma = librosa.segment.recurrence_matrix(chroma, metric='cosine', mode='affinity')

    # compute mel-spec features and ssm
    mel_spec = librosa.feature.melspectrogram(S=S)
    ssm_melspec = librosa.segment.recurrence_matrix(mel_spec, metric='cosine', mode='affinity')

    # compute mfcc from mel-spec
    mfcc = librosa.feature.mfcc(S=mel_spec)
    ssm_mfcc = librosa.segment.recurrence_matrix(mfcc, metric='cosine', mode='affinity')

    return ssm_mfcc, ssm_chroma, ssm_melspec


# compress SSMs into png format
def ssm_to_png(mfcc, chroma, melspec, out_path):
    rgbArray = np.zeros((mfcc.shape[0], mfcc.shape[1], 3), 'uint8')

    rgbArray[..., 0] = (mfcc / mfcc.max()) * 255
    rgbArray[..., 1] = (chroma / chroma.max()) * 255
    rgbArray[..., 2] = (melspec / melspec.max()) * 255

    img = Image.fromarray(rgbArray)
    img.save(out_path)


# return bounding box of chorus, verse, and intro labels of jams dataset
def get_bbox(anno_path):
    # open file
    with open(anno_path) as a:
        data = json.load(a)

        # check through all annotations and count segment labels
    obj_list = []

    for annotation in data['annotations']:

        chorus = []
        verse = []
        intro = []
        segments = []

        for segment in annotation['data']:
            if segment['value'] == "chorus":
                chorus.append([segment['time'], segment["duration"]])
            elif segment['value'] == "verse":
                verse.append([segment['time'], segment["duration"]])
            elif segment['value'] == "intro":
                intro.append([segment['time'], segment["duration"]])

        segments.append(chorus)
        segments.append(verse)
        segments.append(intro)

        colors = ['r', 'g', 'b']
        c = 0
        scale = (44100 / 8192) / 2
        for seglabels in segments:
            segcombos = itertools.product(seglabels, repeat=2)

            # get the segment combination bounding boxes
            for combo in segcombos:
                rect = patches.Rectangle((combo[0][0] * scale, combo[1][0] * scale), combo[0][1] * scale,
                                         combo[1][1] * scale, linewidth=1, edgecolor=colors[c], facecolor='none')
                bound = rect.get_bbox()

                xcount = int(round((bound.x1 - bound.x0) / 0.5))
                ycount = int(round((bound.y1 - bound.y0) / 0.5))
                px = np.linspace(bound.x0, bound.x1, xcount)
                py = np.linspace(bound.y0, bound.y1, ycount)

                poly1 = [(bound.x0, y) for y in py]
                poly2 = [(bound.x1, y) for y in py]
                poly3 = [(x, bound.y0) for x in px]
                poly4 = [(x, bound.y1) for x in px]

                newpoly = np.concatenate((poly1, poly2, poly3, poly4))
                newpoly = np.array(list(np.hstack(newpoly)))
                newpoly = (np.transpose(newpoly.reshape((np.shape(newpoly)[0], 1)))).tolist()

                obj = {
                    "bbox": [bound.x0, bound.y0, bound.x1, bound.y1],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": newpoly,
                    "category_id": c,
                }
                obj_list.append(obj)

            c = c + 1
    return obj_list


# return data in detectron dataset format
def get_audio_dicts(ds_dir):
    annos = '/home/shared/cuzokwe/datasets/SALAMI/references/'
    audio = ds_dir + 'audio'

    dataset_annos = []
    # Parse through available audio tags
    for file in os.listdir(audio):
        anno = annos + 'SALAMI_' + file.split(".")[0] + '.jams'
        # print('audio file: ' + file.split(".")[0])
        if os.path.exists(anno):
            record = {}

            img = ds_dir + "/images/" + file.split(".")[0] + '.png'
            if os.path.exists(img):
                record["file_name"] = img
            else:
                mfcc, chroma, melspec = get_3ssm(audio + '/' + file.split(".")[0] + '.wav')
                ssm_to_png(mfcc, chroma, melspec, img)

                record["file_name"] = img

            record["image_id"] = int(file.split(".")[0])

            img = cv2.imread(img)
            dimensions = img.shape
            record["height"] = dimensions[0]
            record["width"] = dimensions[1]

            objs = get_bbox(anno)
            record["annotations"] = objs

        dataset_annos.append(record)
    return dataset_annos