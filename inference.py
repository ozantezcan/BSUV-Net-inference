import sys
sys.path.append("./utils/segmentation/") # Add the projects root dir to paths
import configs.infer_config as cfg
import cv2
import torch
from utils.data_loader import videoLoader
from utils import seg_utils as seg
import time
import numpy as np

assert len(sys.argv) > 2, "You must provide the input and output video paths"
inp_path = sys.argv[1]
out_path = sys.argv[2]

# Load the segmentation network
seg_network = seg.segModel(cfg.SemanticSegmentation.yaml_path,
                           cfg.SemanticSegmentation.encoder_path,
                           cfg.SemanticSegmentation.decoder_path)



# Start output video
vid_loader = videoLoader(inp_path, empty_bg="no",
                         recent_bg=50, seg_network=seg_network,
                         transforms_pre=cfg.BSUVNet.transforms_pre, transforms_post=cfg.BSUVNet.transforms_post)
fr = next(iter(vid_loader))
c, h, w = fr.size()
h = int(16 * int(h / 16))
w = int(16 * int(w / 16))
vid_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MP4V"), 30, (3*w+20, h), isColor=True)


# Start Video Loader
vid_loader = videoLoader(inp_path, empty_bg=cfg.BSUVNet.emtpy_bg, empty_win_len=cfg.BSUVNet.empty_win_len,
                         recent_bg=cfg.BSUVNet.recent_bg, seg_network=seg_network,
                         transforms_pre=cfg.BSUVNet.transforms_pre, transforms_post=cfg.BSUVNet.transforms_post)
tensor_loader = torch.utils.data.DataLoader(dataset=vid_loader, batch_size=1)

# Load BSUV-Net
bsuvnet = torch.load(cfg.BSUVNet.model_path)
bsuvnet.cuda().eval()

# Start Inference
num_frames = 0
start = time.time()  # Inference start time
with torch.no_grad():
    for inp in tensor_loader:
        num_frames += 1
        bgs_pred = bsuvnet(inp.cuda().float()).cpu().numpy()[0, 0, :, :]

        # Construct the output frame
        inp_org = inp.numpy()[0, -3:, :, :] * np.asarray(cfg.BSUVNet.std_rgb).reshape(3, 1, 1) + \
                  np.asarray(cfg.BSUVNet.mean_rgb).reshape(3, 1, 1)
        inp_org[inp_org < 0] = 0
        inp_org[inp_org > 1] = 1
        inp_org = inp_org.transpose(1, 2, 0)[:, :, ::-1]

        fr = np.ones((h, 3 * w + 20, 3)) * 0.5
        fr[:, :w, :] = inp_org
        for ch in range(3):
            fr[:, w + 10:2 * w + 10, ch] = bgs_pred
            fr[:, 2 * w + 20:, ch] = inp_org[:, :, ch] * (bgs_pred + 1) / 2

        vid_out.write((fr * 255).astype(np.uint8))
        if num_frames % 100 == 0:
            print("%d frames completed" %num_frames)
vid_out.release()
end = time.time()  # Inference end time
fps = num_frames / (end - start)
print("%.3f FPS for (%d, %d) resolution" % (fps, w, h))
