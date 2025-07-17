import os
import time
from os import path
import argparse
import numpy as np
import torch
import cv2
import imageio.v2 as io
from network import TriSegNet

import subprocess
import tempfile
import os
import Imath
import OpenEXR

def save_exr_rgb_float(path, rgb_array):
    height, width = rgb_array.shape[:2]
    header = OpenEXR.Header(width, height)

    # 定义通道类型为 FLOAT
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {'R': pt, 'G': pt, 'B': pt}

    # 转成 byte 数据
    R = rgb_array[:, :, 0].astype(np.float32).tobytes()
    G = rgb_array[:, :, 1].astype(np.float32).tobytes()
    B = rgb_array[:, :, 2].astype(np.float32).tobytes()

    # 写入 EXR 文件
    exr_file = OpenEXR.OutputFile(path, header)
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()

def save_avif_hdr_bt2020(prediction, out_name):
    # 创建临时 EXR 中间文件（保留 HDR float32 数据）
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp_exr:
        tmp_exr_path = tmp_exr.name

        # 保存为 EXR（HDR 容器）
        # import imageio.v3 as iio
        # iio.imwrite(tmp_exr_path, prediction.astype(np.float32), format="EXR-FI")
        save_exr_rgb_float(tmp_exr_path, prediction)

    # 构建 avifenc 命令：使用 BT.2020 色彩、PQ gamma、YUV420编码
    avif_command = [
        "avifenc",
        tmp_exr_path, out_name,
        "--min 0", "--max 1",                         # HDR 范围（线性）
        "--colorPrimaries", "9",                     # BT.2020
        "--transferCharacteristics", "16",           # PQ（HDR gamma）
        "--matrixCoefficients", "9",                 # BT.2020 matrix
        "--depth", "10",                             # 10位编码
        "--yuv", "420",                              # 色度采样
        "--quality", "100", "--speed", "6"           # 编码质量与速度
    ]

    print("正在编码 HDR AVIF（BT.2020） →", out_name)
    subprocess.run(" ".join(avif_command), shell=True)

    os.remove(tmp_exr_path)  # 清理临时文件


### System utilities ###
def process_path(directory, create=False):
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


def compose(transforms):
    """Composes list of transforms (each accept and return one item)"""
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), "list of functions expected"

    def composition(obj):
        """Composite function"""
        for transform in transforms:
            obj = transform(obj)
        return obj
    return composition


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


### Image utilities ###
def np2torch(img, from_bgr=True):
    img = img[:, :, [2, 1, 0]] if from_bgr else img
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float().float()#.half()


def torch2np(t_img, to_bgr=True):
    img_np = t_img.numpy()  # t_img.detach().numpy()
    out = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)).astype(np.float32) if to_bgr \
        else np.transpose(img_np, (1, 2, 0)).astype(np.float32)
    return out


def resize(x):
    h, w = x.shape[:2]

    # 自动调整为可被8整除的尺寸（适配2x下采样3次）
    h_new = h - h % 8
    w_new = w - w % 8

    # 如果尺寸已经符合，就不变；否则执行 resize
    if h_new != h or w_new != w:
        print(f"自动调整图像尺寸: 原始 ({h}, {w}) -> 调整为 ({h_new}, {w_new})")
        x = cv2.resize(x, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    return x


class Exposure(object):
    def __init__(self, stops, gamma):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img*(2**self.stops), 0, 1)**self.gamma


### Parameters ###
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('ldr', nargs='+', type=process_path, help='Ldr image(s)')
arg('-out', type=lambda x: process_path(x, True), default=None, help='Output location.')
arg('-resize', type=str2bool, default=False, help='Use resized input.')
arg('-width', type=int, default=1920, help='Image width resizing.')
arg('-height', type=int, default=1080, help='Image height resizing.')
arg('-tag', default=None, help='Tag name for outputs.')
arg('-use_gpu', type=str2bool, default=torch.cuda.is_available(), help='Use GPU for prediction.')
arg('-in_bitdepth', type=int, default=8, help='Bit depth of input SDR frames.')
arg('-out_format', choices=['tif', 'exr', 'png'], default='tif', help='Encapsulation of output HDR frames.')
opt = parser.parse_args()


### Load network ###
net = TriSegNet().float() #.half()
net.load_state_dict(torch.load('method/params_3DM.pth', map_location=lambda s, l: s))

### Defined Pre-process ##
preprocess = compose([lambda x: x.astype('float32'), resize])

### Loading single frames ###
for ldr_file in opt.ldr:
    loaded = cv2.imread(ldr_file, flags=cv2.IMREAD_UNCHANGED) / (2.0 ** opt.in_bitdepth - 1.0)
    print('Could not load {0}'.format(ldr_file)) if loaded is None else print('Image {0} loaded!'.format(ldr_file))
    start = time.time()
    ldr_input = preprocess(loaded)
    ldr_input = np2torch(ldr_input, from_bgr=True).unsqueeze(dim=0)

    if opt.use_gpu:
        net.cuda()
        ldr_input = ldr_input.cuda()

    with torch.no_grad():
        prediction = net(ldr_input).detach()[0].float().cpu()
        prediction = torch2np(prediction, to_bgr=False)

    if opt.out_format == 'tif':
        out_name = create_name(ldr_file, 'HDR', 'tif', opt.out, opt.tag)
        prediction = np.round(prediction * 65535.0).astype(np.uint16)
        io.imwrite(out_name, prediction)
        # cv2.imwrite(out_name, prediction, (int(cv2.IMWRITE_TIFF_COMPRESSION), 5))
        # flag 5 means LZW compression, see: opencv\sources\3rdparty\libtiff\tiff.h
    elif opt.out_format == 'exr':
        out_name = create_name(ldr_file, 'HDR', 'exr', opt.out, opt.tag)
        print("保存 EXR 到：", out_name)
        MIN_SIZE = 64
        h, w = prediction.shape[:2]
        if h < MIN_SIZE or w < MIN_SIZE:
            print(f"图像太小，自动扩展为 ({MIN_SIZE}, {MIN_SIZE})")
            prediction = cv2.resize(prediction, (MIN_SIZE, MIN_SIZE), interpolation=cv2.INTER_LINEAR)
        RGB = prediction.astype(np.float32)  # shape: (H, W, 3)
        channels = { "RGB": RGB }
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage
        }
        with OpenEXR.File(header, channels) as exr_out:
            exr_out.write(out_name)
    elif opt.out_format == 'png':
        out_name = create_name(ldr_file, 'HDR', 'png', opt.out, opt.tag)
        prediction = np.round(prediction * 65535.0).astype(np.uint16)
        io.imwrite(out_name, prediction)
        # cv2.imwrite(out_name, prediction)
    else:
        raise AttributeError('Unsupported output format!')

    end = time.time()
    print('Finish processing {0}. \n takes {1} seconds. \n -------------------------------------'
          ''.format(ldr_file, '%.04f' % (end-start)))
