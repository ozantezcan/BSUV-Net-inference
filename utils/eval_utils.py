import numpy as np
import csv
import torch
import torchvision
from utils.losses import getValid
from utils.data_loader import CDNet2014Loader
from utils import augmentations as aug
# Locations of each video in the CSV file
csv_header2loc = {'len': 160, 'highway': 1, 'pedestrians': 4, 'office': 7, 'PETS2006': 10, 'badminton': 13, 'traffic': 16,
                  'boulevard': 19, 'sidewalk': 22, 'skating': 25, 'blizzard': 28, 'snowFall': 31, 'wetSnow': 34, 'boats': 37,
                  'canoe': 40, 'fall': 43, 'fountain01': 46, 'fountain02': 49, 'overpass': 52, 'abandonedBox': 55,
                  'parking': 58, 'sofa': 61, 'streetLight': 64, 'tramstop': 67, 'winterDriveway': 70,
                  'port_0_17fps': 73, 'tramCrossroad_1fps': 76, 'tunnelExit_0_35fps': 79, 'turnpike_0_5fps': 82,
                  'bridgeEntry': 85, 'busyBoulvard': 88, 'fluidHighway': 91, 'streetCornerAtNight': 94, 'tramStation': 97,
                  'winterStreet': 100, 'continuousPan': 103, 'intermittentPan': 106, 'twoPositionPTZCam': 109,
                  'zoomInZoomOut': 112, 'backdoor': 115, 'bungalows': 118, 'busStation': 121, 'copyMachine': 124,
                  'cubicle': 127, 'peopleInShade': 130, 'corridor': 133, 'diningRoom': 136, 'lakeSide': 139, 'library': 142,
                  'park': 145, 'turbulence0': 148, 'turbulence1': 151, 'turbulence2': 154, 'turbulence3': 157}

def evalVideo(cat, vid, model, empty_bg=False, recent_bg=False, segmentation_ch=False, eps=1e-5, adversary="no"):
    """ Evalautes the trained model on all ROI frames of cat/vid
    Args:
        :cat (string):                  Category
        :video (string):                Video
        :model (torch model):           Trained PyTorch model
        :empty_bg (boolean):            Boolean for using the empty background frame
        :recent_bg (boolean):           Boolean for using the recent background frame
        :segmentation_ch (boolean):     Boolean for using the segmentation maps
        :eps (float):                   A small multiplier for making the operations easier
        :adversary (str):               "no": No adversarial part
                                        "dann": Domain adversarial neural network
    """

    transforms = [
        aug.ToTensor(),
        aug.NormalizeTensor(mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
                            mean_seg=[0.5], std_seg=[0.5], segmentation_ch=segmentation_ch)
    ]
    dataloader = CDNet2014Loader({cat:[vid]}, empty_bg=empty_bg, recent_bg=recent_bg,
                              segmentation_ch=segmentation_ch, transforms=transforms,
                              use_selected=False, empty_bg_radnomize=False)
    tensorloader = torch.utils.data.DataLoader(dataset=dataloader,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    model.eval() # Evaluation mode
    tp, fp, fn = 0, 0, 0
    for i, data in enumerate(tensorloader):
        if (i+1) % 100 == 0:
            print("%d/%d" %(i+1, len(tensorloader)))
        input, label = data[0].float(), data[1].float()
        """TO-DO!
        Do not use .cuda() here. it is not modular. Try to make these more modular
        """
        input, label = input.cuda(), label.cuda()
        if adversary == "no":
            output = model(input)
        elif adversary == "dann":
            output, _ = model(input)
        elif adversary == "agnostic_dann":
            output, _, _ = model(input, alpha=[0, 0])
        label_1d, output_1d = getValid(label, output)
        #
        tp += eps * torch.sum(label_1d * output_1d).item()
        fp += eps * torch.sum((1-label_1d) * output_1d).item()
        fn += eps * torch.sum(label_1d * (1-output_1d)).item()

    # Calculate the statistics
    prec = tp / (tp + fp) if tp + fp > 0 else float('nan')
    recall = tp / (tp + fn) if tp + fn > 0 else float('nan')
    f_score = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else float('nan')
    return 1-recall, prec, f_score

def logVideos(dataset, model, model_name, csv_path, empty_bg=False, recent_bg=False, segmentation_ch=False, eps=1e-5, adversary="no"):
    """ Evaluate the videos given in dataset and log them to a csv file
    Args:
        :dataset (dict):                Dictionary of dataset. Keys are the categories (string),
                                        values are the arrays of video names (strings).
        :model (torch model):           Trained PyTorch model
        :model_name (string):           Name of the model for logging
        :csv_path (string):             Path to the CSV file
        :empty_bg (boolean):            Boolean for using the empty background frame
        :recent_bg (boolean):           Boolean for using the recent background frame
        :segmentation_ch (boolean):     Boolean for using the segmentation maps
        :eps (float):                   A small multiplier for making the operations easier
        :adversary (str):               "no": No adversarial part
                                        "dann": Domain adversarial neural network
    """

    new_row = [0] * csv_header2loc['len']
    new_row[0] = model_name

    for cat, vids in dataset.items():
        for vid in vids:
            print(vid)
            fnr, prec, f_score = evalVideo(cat, vid, model, empty_bg=empty_bg, recent_bg=recent_bg,
                                           segmentation_ch=segmentation_ch, eps=eps, adversary=adversary)

            new_row[csv_header2loc[vid]] = fnr
            new_row[csv_header2loc[vid]+1] = prec
            new_row[csv_header2loc[vid]+2] = f_score


    with open(csv_path, mode='a') as log_file:
        employee_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(new_row)

    print('Done!!!')
