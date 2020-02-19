import os
import numpy as np
import torch.utils.data as data
from utils import seg_utils as seg
import cv2
import queue

class videoLoader(data.IterableDataset):
    """
    Video loader class
    creates a data loader form a video
    """

    def __init__(self, vid_path, empty_bg="no", empty_win_len=0, empty_bg_path="",
                 recent_bg=0, recent_bg_opp="mean", seg_network=None,
                 transforms_pre=None, transforms_post=None):
        """
        Initialization of video loader
        Args:
            vid_path (str):   Path to the input video
            empty_bg (str):     "no" : No empty background
                                "manual": Manually specified empty background by <empty_bg_path>
                                "automatic": Use the median first <empty_win_len> as empty background
            empty_win_len (int):Number of initial frames to be medianed for empty background. 0 means median of all frames
                                Used only when empty_bg = "manual"
            empty_bg_path (str):Path of the empty background
                                Used only when empty_bg = "automatic"
            recent_bg (int):    Number of last frames to be averaged for recent background. 0 means no recent background
            recent_bg_opp (str): The averaging operation for recent background. Only "mean" for now
            seg_network:        Segmentation module to be used for semantic segmentation
            transforms_pre (torchvision.transforms): Transforms to be applied to each input before converting to tensors
            transforms_pre (torchvision.transforms): Transforms to be applied to each input after converting to tensors
        """

        # Start the video reader
        self.vid_cap = cv2.VideoCapture(vid_path)
        assert self.vid_cap.isOpened(), "Error opening video stream or file: {}".format(vid_path)

        # Initialize the segmentation network
        self.seg = False if seg_network is None else True
        if self.seg:
            self.seg_network = seg_network

        # Initialize empty background
        self.empty_bg = None
        if empty_bg == "manual":
            self.empty_bg = self.__readRGB(empty_bg_path)

        elif empty_bg == "automatic":
            # For time efficiency compute the median of medians

            med_arr = []
            fr_arr = []
            win_size = 100
            fr_counter = 0

            while self.vid_cap.isOpened():
                # Capture frame-by-frame
                ret, fr = self.vid_cap.read()
                if ret:
                    # Preprocess
                    fr = self.__preProc(fr)
                    fr_arr.append(fr)

                    if len(fr_arr) == win_size:
                        med_arr.append(np.median(np.asarray(fr_arr), axis=0))
                        fr_arr = []

                    fr_counter += 1
                    if fr_counter == empty_win_len:
                        break
                # Break the loop
                else:
                    break

            if len(fr_arr) > 0:
                med_arr.append(np.median(np.asarray(fr_arr), axis=0))
            self.empty_bg = np.median(np.asarray(med_arr), axis=0)
            print("Empty background is completed")
            self.vid_cap = cv2.VideoCapture(vid_path)

        # Initialize the segmentation of empty background
        self.empty_bg_seg = None


        # Initialize an array of recent background
        self.recent_bg = recent_bg
        self.recent_bg_opp = recent_bg_opp
        assert recent_bg or recent_bg_opp in ["mean"], \
            "{} is not defined for <recent_bg_opp>. Use 'mean'.".format(recent_bg_opp)

        ret, fr = self.vid_cap.read()
        if self.recent_bg:
            self.recent_bg_sum = np.zeros_like(fr).astype(np.float)

        # Reinitialize the video reader
        self.vid_cap = cv2.VideoCapture(vid_path)
        self.transforms_pre = transforms_pre
        self.transforms_post = transforms_post


    def __iter__(self):

        # Array initialization for recent background
        if self.recent_bg:
            recent_bg_arr = queue.Queue(maxsize=self.recent_bg)

        # Read until video is completed
        while self.vid_cap.isOpened():
            # Capture frame-by-frame
            ret, fr = self.vid_cap.read()
            if ret:
                # Preprocess
                fr = self.__preProc(fr)

                # Initialize input
                inp = {"empty_bg_seg": self.empty_bg_seg, "empty_bg": self.empty_bg,
                       "recent_bg_seg": None, "recent_bg": None,
                       "current_fr_seg": None, "current_fr": None}

                # Current frame
                inp["current_fr"] = fr

                # Recent background
                if self.recent_bg:
                    if recent_bg_arr.full():
                        old_bg = recent_bg_arr.get()
                        if self.recent_bg_opp == "mean":
                            self.recent_bg_sum -= old_bg

                    if self.recent_bg_opp == "mean":
                        self.recent_bg_sum += inp["current_fr"]
                        recent_bg_arr.put(inp["current_fr"])
                        inp["recent_bg"] = self.recent_bg_sum / recent_bg_arr.qsize()


                for transform in self.transforms_pre:
                    inp, _ = transform(inp, fr[:, :, :1])

                if self.seg:
                    if inp["empty_bg_seg"] is None:
                        self.empty_bg_seg = seg.getFPM(inp["empty_bg"], self.seg_network)
                        inp["empty_bg_seg"] = self.empty_bg_seg
                    inp["recent_bg_seg"] = seg.getFPM(inp["recent_bg"], self.seg_network)
                    inp["current_fr_seg"] = seg.getFPM(inp["current_fr"], self.seg_network)

                for transform in self.transforms_post:
                    inp, _ = transform(inp, fr[:, :, :1])

                yield inp
            # Break the loop
            else:
                break

    def __len__(self):
        return self.n_data

    def __readRGB(self, path):
        assert os.path.exists(path), "{} does not exist".format(path)
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float)/255
        h, w, _ = im.shape
        h_valid = int(h / 16) * 16
        w_valid = int(w / 16) * 16
        return im[:h_valid, :w_valid, :]

    def __preProc(self, fr):
        h, w, _ = fr.shape
        h_valid = int(h / 16) * 16
        w_valid = int(w / 16) * 16
        return cv2.cvtColor(fr[:h_valid, :w_valid, :], cv2.COLOR_BGR2RGB).astype(np.float)/255

