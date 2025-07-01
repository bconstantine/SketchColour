import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

from informative_drawings.base.model import Generator, GlobalGenerator2, InceptionV3
from informative_drawings.base.utils import channel2width
import cv2
from typing import Union, List
import subprocess

from tqdm import tqdm

# ─── tiny helper ──────────────────────────────────────────────────────────────
def ensure_dir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_params():
    new_h = new_w = 256 

    x = random.randint(0, np.maximum(0, 0))
    y = random.randint(0, np.maximum(0, 0))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, norm=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if not grayscale:
            if norm:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_modified(opt,
                  params=None,
                  grayscale=False,
                  method=Image.BICUBIC,
                  convert=True,
                  norm=True,
                  do_crop=False):       # <── new arg
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    # resize to the generator’s expected size
    osize = [512, 512]                # keep this whatever your net expects
    transform_list.append(transforms.Resize(osize, method))

    # ── remove crop/flip when do_crop=False ──────────────────────────
    if do_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list.append(transforms.ToTensor())
        if not grayscale and norm:
            transform_list.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    color = (255, 255, 255)
    if img.mode == 'L':
        color = (255)
    elif img.mode == 'RGBA':
        color = (255, 255, 255, 255)

    if (ow > tw and oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    elif ow > tw:
        ww = img.crop((x1, 0, x1 + tw, oh))
        return add_margin(ww, size, 0, (th-oh)//2, color)
    elif oh > th:
        hh = img.crop((0, y1, ow, y1 + th))
        return add_margin(hh, size, (tw-ow)//2, 0, color)
    return img

def add_margin(pil_img, newsize, left, top, color=(255, 255, 255)):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (newsize, newsize), color)
    result.paste(pil_img, (left, top))
    return result


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    return any(filename.endswith(extension) for extension in extensions)

def is_video_file(filename):
    extensions = ['.mp4']
    return any(filename.endswith(extension) for extension in extensions)

def make_dataset(data_dir):
    images = []
    assert os.path.isdir(data_dir), '%s is not a valid directory' % data_dir
    for root, _, fnames in sorted(os.walk(data_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class UnpairedDepthDataset(data.Dataset):
    def __init__(self, root, root2, transforms_r=None, mode='train', depthroot=''):

        self.root = root
        self.mode = mode

        all_img = make_dataset(self.root)

        self.depth_maps = 0

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)
        
        if mode == 'train':
            
            self.img2 = make_dataset(root2)

            if len(self.data) > len(self.img2):
                howmanyrepeat = (len(self.data) // len(self.img2)) + 1
                self.img2 = self.img2 * howmanyrepeat
            elif len(self.img2) > len(self.data):
                howmanyrepeat = (len(self.img2) // len(self.data)) + 1
                self.data = self.data * howmanyrepeat
                self.depth_maps = self.depth_maps * howmanyrepeat
            

            cutoff = min(len(self.data), len(self.img2))

            self.data = self.data[:cutoff] 
            self.img2 = self.img2[:cutoff] 

            self.min_length =cutoff
        else:
            self.min_length = len(self.data)


    def __getitem__(self, index):

        img_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split('.')[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params()
        A_transform = get_transform(transform_params, grayscale=False, norm=False)
        B_transform = get_transform(transform_params, grayscale=True, norm=False)        

        if self.mode != 'train':
            A_transform = self.transform_r

        img_r = A_transform(img_r )

        B_mode = 'L'

        img_depth = 0

        img_normals = 0
        label = 0

        input_dict = {'r': img_r, 'depth': img_depth, 'path': img_path, 'index': index, 'name' : base, 'label': label}

        if self.mode=='train':
            cur_path = self.img2[index]
            cur_img = B_transform(Image.open(cur_path).convert(B_mode))
            input_dict['line'] = cur_img

        return input_dict

    def __len__(self):
        return self.min_length



def run_inference(input_dir, output_dir, threshold):
    with torch.no_grad():
        lower = torch.tensor(0.0, device='cuda')
        upper = torch.tensor(1.0, device='cuda')
        net_G = 0
        net_G = Generator(3, 1, 3)
        net_G.cuda()

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base', 'checkpoints', 'anime_style', 'netG_A_latest.pth')

        net_G.load_state_dict(torch.load(model_path))
        net_G.eval()

        input_full_dir = os.path.join(os.getcwd(), input_dir)
        output_full_dir = os.path.join(os.getcwd(), output_dir)

        transforms_r = [transforms.Resize(int(512), Image.BICUBIC),
                       transforms.ToTensor()]

        test_data = UnpairedDepthDataset(input_full_dir, '', transforms_r=transforms_r, 
                    mode='test', depthroot='')

        dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        for i, batch in enumerate(dataloader):
            img_r  = Variable(batch['r']).cuda()
            img_depth  = Variable(batch['depth']).cuda()
            real_A = img_r

            name = batch['name'][0]
            
            input_image = real_A
            image = net_G(input_image)
            if threshold > 0:
                binarized = torch.where(image > threshold, upper, lower)
                save_image(binarized.data, output_full_dir+'/%s_out.png' % name)
            else:
                save_image(image.data, output_full_dir+'/%s_out.png' % name)

            sys.stdout.write('\rGenerated images %04d of %04d' % (i, len(dataloader)))

        sys.stdout.write('\n')
# ─── (renamed) legacy single-image path ──────────────────────────────────────
def run_inference_image(image_list: List[Path],
                        in_root: Path,
                        out_root: Path,
                        threshold: float):
    """Original logic, trimmed to accept an explicit file list."""
    with torch.no_grad():
        lower, upper = torch.tensor(0.).cuda(), torch.tensor(1.).cuda()

        net_G = Generator(3, 1, 3).cuda()
        ckpt = '/gpfs/junlab/wangwanding24/KeyMotion/weights/sketch_generation/netG_A_latest.pth'
        net_G.load_state_dict(torch.load(ckpt))
        net_G.eval()

        params = get_params()
        A_transform = get_transform_modified(opt=params, params=params,
                                    grayscale=False, norm=False, 
                                    do_crop=False)

        for idx, img_path in enumerate(image_list, 1):
            rel_path = img_path.relative_to(in_root)
            out_dir  = out_root / rel_path.parent
            ensure_dir(out_dir)

            base = img_path.stem
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            inp = A_transform(img).unsqueeze(0).cuda()
            pred = net_G(inp)
            
            pred = F.interpolate(pred,
                                    size=(orig_h, orig_w),
                                    mode="bilinear",
                                    align_corners=False)

            if threshold > 0:
                pred = torch.where(pred > threshold, upper, lower)
            # resize to the original frame size
            
            save_image(pred, out_dir / f"{base}sketch.png")
            sys.stdout.write(f"\rGenerated {idx}/{len(image_list)}")
        sys.stdout.write("\n")

# ─── NEW video pipeline ───────────────────────────────────────────────────────
def run_inference_video(video_paths: List[Path],
                        in_root: Path,
                        out_root: Path,
                        threshold: float):
    with torch.no_grad():
        lower, upper = torch.tensor(0.).cuda(), torch.tensor(1.).cuda()

        net_G = Generator(3, 1, 3).cuda()
        ckpt = 'weights/sketch_generation/netG_A_latest.pth'
        net_G.load_state_dict(torch.load(ckpt))
        net_G.eval()

        # fixed transforms (same as your test pipeline, but params keyworded)
        params = get_params()
        A_transform = get_transform_modified(opt=params, params=params,
                                    grayscale=False, norm=False, do_crop = False)

        for vid_idx, vid_path in tqdm(enumerate(video_paths, 1)):
            rel_path = vid_path.relative_to(in_root)               # 42/003/clip.mp4
            out_dir  = out_root / rel_path.parent                  # …/sketch/42/003
            ensure_dir(out_dir)

            temp_out_name = vid_path.stem + '_temp.avi'
            temp_out_file = str(out_dir / temp_out_name)
            out_name = vid_path.stem + "_sketch.mp4"
            out_file = str(out_dir / out_name)

            cap = cv2.VideoCapture(str(vid_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                temp_out_file,
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (W, H),  # keep colour for simplicity
                True
            )
            if not writer.isOpened():
                print(f"Error: Could not open temporary AVI writer for {temp_out_file}")
                cap.release()
                return

            frame_no = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                # BGR ➜ RGB ➜ PIL
                img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                orig_w, orig_h = img_pil.size

                input_tensor = A_transform(img_pil).unsqueeze(0).cuda()
                pred = net_G(input_tensor)
                pred = F.interpolate(pred,
                                    size=(orig_h, orig_w),
                                    mode="bilinear",
                                    align_corners=False)
                
                if threshold > 0:
                    pred = torch.where(pred > threshold, upper, lower)
                out_np = pred[0,0]
                out_np = (out_np * 255).clamp(0, 255)  # scale to 0-255
                out_np = out_np.byte().cpu().numpy()
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_GRAY2BGR)
                writer.write(out_bgr)

                frame_no += 1
                # sys.stdout.write(f"\r[{vid_idx}/{len(video_paths)}] "
                #                  f"frames processed: {frame_no}")

            cap.release()
            writer.release()
            # 5) Re-encode to H.264 MP4 silently
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_out_file,
                '-c:v', 'libx264',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-movflags', 'faststart',
                out_file
            ]
            subprocess.run(ffmpeg_cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

            os.remove(temp_out_file)
            sys.stdout.write("\n")

def main(args):
    # ─── modified main() ──────────────────────────────────────────────────────────
    print(torch.cuda.is_available())

    dataroot = Path(args.dataroot)          # e.g. dataset/a/keyframe
    outroot  = Path(args.output_folder)     # e.g. dataset/a/sketch
    thr      = float(args.binarize_threshold) / 255.0

    video_list, image_list = [], []

    for fp in dataroot.rglob("*"):
        if fp.suffix.lower() == ".mp4":
            video_list.append(fp)
        elif fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_list.append(fp)

    if image_list and args.input_type=="image":
        run_inference_image(image_list, dataroot, outroot, thr)   # unchanged logic

    if video_list and args.input_type=="video":
        run_inference_video(video_list, dataroot, outroot, thr)   # ← NEW

