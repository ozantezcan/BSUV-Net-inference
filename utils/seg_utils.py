from utils.segmentation.config import cfg
from utils.segmentation.models import ModelBuilder, SegmentationModule
from torch import from_numpy, no_grad
import torch.nn as nn
import torchvision.transforms as tvtf
import numpy as np

def probs2fg(seg_map):
    """
    Calculates the Foreground Probability Map from segmentation output
    Args:
        seg_map (numpy array of size (W, H, 150)):  Array of pixelwise semantic segmentation probabilities fro 150
                                                    classes of ADE20K dataset
                                                    (https://groups.csail.mit.edu/vision/datasets/ADE20K/)

    Returns:
        numpy array of size (W, H): Sum of foreground class probabilities as defined in the paper
                                    (https://arxiv.org/abs/1907.11371)
    """
    fg_objects = [12, 20, 39, 41, 67, 76, 80, 83, 98, 102, 115, 127]
    fg_probs = np.sum(seg_map[:, :, fg_objects], axis=-1)
    return fg_probs

def segModel(yaml_path, encoder_path, decoder_path):
    """
    Load the semantic segmentation model
    Args:
        yaml_path (str):    Path of the model's config file
        encoder_path (str): Path of the encoder weights file
        decoder_path (str): Path of the decoder weights file

    Returns:
        segmentation module
    """
    cfg.merge_from_file(yaml_path)
    cfg.MODEL.weights_encoder = encoder_path
    cfg.MODEL.weights_decoder = decoder_path

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    segmentation_module.eval()
    return segmentation_module

def getFPM(im, seg_network, fpm=True):
    """
    Returns the Foreground Probability Map (FPM) of the input tensor
    Args:
        im (numpy array of size (W, H, 3)): input image in float32 format
        seg_network (segmentation module):  segmentation module
        fpm (boolean): Return FPM if true, else return segmentation outcome
    Returns:
        numpy array of size (W, H, 1): FPM of the input image
    """
    if im is None:  # trivial case
        return None

    normalize = tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im_tensor = normalize(from_numpy(im.transpose(2, 0, 1))).unsqueeze(0).cuda().float()
    h, w, _ = im.shape
    with no_grad():
        seg_map = seg_network({'img_data': im_tensor}, segSize=[h, w]).squeeze(0).cpu().numpy()
    
    if fpm:
        return np.expand_dims(probs2fg(seg_map.transpose(1, 2, 0)).astype(np.float), -1)
    else:
        return seg_map

