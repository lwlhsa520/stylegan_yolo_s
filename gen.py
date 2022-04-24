import os
import time

import PIL

import legacy
import torch
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


import dnnlib

# BOX_COLOR = (255, 0, 0)  # Red
# TEXT_COLOR = (255, 255, 255)  # White
from utils.general import xyxy2xywh, xywh2xyxy, clip_coords, increment_path


def visualize_bbox(imgs, bboxs, thickness=1):
    """Visualizes a single bounding box on the image"""
    for idx, (img, box) in enumerate(zip(imgs, bboxs)):
        img = img.to(torch.int8).cpu().numpy()
        for bbox in box:
            _, x_c, y_c, w, h = bbox * 255
            x_min, x_max, y_min, y_max = int(x_c - w/2), int(x_c + w/2), int(y_c - w/2), int(y_c + h/2)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0))

            #  ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
            # cv2.putText(
            #     img,
            #     #text=class_name,
            #     #org=(x_min, y_min - int(0.3 * text_height)),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.35,
            #     color=TEXT_COLOR,
            #     lineType=cv2.LINE_AA,
            # )
        cv2.imwrite(f"gen_{idx}.png", img)

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def label_preparetion(labels, nums = 256):
    new_bbox, new_label = [], []
    for batch, label in enumerate(labels):
        if label.shape[0] < nums:
            tmp = torch.zeros([nums-label.shape[0],5])
            label[:, 0] = 1
            label = torch.cat([label, tmp], dim=0)

        new_bbox.append(label[:nums])

    return np.stack(new_bbox)


def generate_labels(batch_size=8, seed=0, nums = 256):
    new_labels = []
    np.random.seed(seed)
    for idx in range(batch_size):
        rnd = np.random.randint(20, 40)
        rnd_level = np.random.choice(np.linspace(1, 4, 31), 1)
        batch = int(800 / (rnd_level * rnd))
        init_ = np.random.choice(np.linspace(20, 200, 181), batch * 2).reshape(batch, 2) / 255
        shape_ = (np.random.choice(np.linspace(-3, 3, 7), batch * 2).reshape(batch, 2) + rnd) / 255
        head_ = np.zeros([batch, 2])
        head_[:, 0] = idx
        new_labels.append(torch.from_numpy(np.concatenate([head_, init_, shape_], -1)))

    return new_labels

def generate_labels2(batch_size=8, seed=0, nums = 256):
    new_labels = []
    np.random.seed(seed)
    bbox = np.array([[0, 0, 130, 130, 40, 40]])
    for _ in range(batch_size):
        new_labels.append(torch.from_numpy(bbox / 255))

    return new_labels

def plot(label, batch, run_dir):
    print("generate groundTrue bbox...")
    gt_img = np.ones([batch, 256, 258, 3]) * 255
    gt_img[:, :, 257:, ] = 0
    for idx, bboxs in enumerate(label):
        for bbox in bboxs:
            (x, y, w, h) = bbox[1:] * 255
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(gt_img[idx], (x1,y1),(x2,y2), (255, 0, 255), 2)

    gt_img = np.concatenate(gt_img, 1)
    cv2.imwrite(run_dir, gt_img)


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    return crop

if __name__ == "__main__":
    batch_gpu = 16
    gh, gw = 4, 4
    seed = 7
    # _label = generate_labels(batch_gpu, seed=10)

    # print(_label)
    _label = np.loadtxt('./data_1_4.txt')
    print(_label)
    label_int = _label.copy()
    label_int[:, 1] = _label[:, 1] - _label[:, 3] / 2
    label_int[:, 2] = _label[:, 2] - _label[:, 4] / 2
    label_int[:, 3] = _label[:, 1] + _label[:, 3] / 2
    label_int[:, 4] = _label[:, 2] + _label[:, 4] / 2

    label_int = (label_int * 255).astype(int)
    print(label_int)
    new_label = []

    new_label.append(torch.from_numpy(_label))

    plot(new_label, batch_gpu, 'gt_label_init.png')

    device = torch.device('cuda')
    gen_label = torch.from_numpy(label_preparetion(new_label)).to(device)

    network_pkl = '../res/stylegan_ada_debug/00091--auto1-batch8/network-snapshot-000800.pkl'
    # network_pkl = '../res/stylegan_ada_debug/00065--auto1-batch8/network-snapshot-004600.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    truncation_psi = 1
    noise_mode = 'const'

    z = torch.from_numpy(np.random.RandomState(5000).randn(1, G.z_dim)).to(device)
    img = G(z, gen_label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    save_image_grid(img.cpu().numpy(), f'gen.png', drange=[-1, 1], grid_size=(1, 1))
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.unbind(0)[0].cpu().numpy()

    # print(label_int[1][1:])
    for idx, label in enumerate(label_int):
        crop = save_one_box(label_int[idx][1:], img)
        cv2.imwrite(f'crop/sc_{idx}.png', crop)

    for lab in label_int:
        cv2.rectangle(img, (lab[1], lab[2]), (lab[3], lab[4]), (255, 0, 0))

    cv2.imwrite("gen_bbox_5000_4.png", img)