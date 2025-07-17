import os
import argparse
import torch
from network import TriSegNet

def export_onnx(output_path, input_height, input_width, use_gpu=False):
    model = TriSegNet().float()
    model.load_state_dict(torch.load('method/params.pth', map_location='cpu'))
    model.eval()

    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.randn(1, 3, input_height, input_width).float().cuda()
    else:
        dummy_input = torch.randn(1, 3, input_height, input_width).float()

    out_name = os.path.abspath(output_path)

    torch.onnx.export(
        model,
        dummy_input,
        out_name,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"✅ ONNX 模型已保存到: {out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='TriSegNet.onnx', help='输出 ONNX 文件路径')
    parser.add_argument('--height', type=int, default=1080, help='输入图像高度')
    parser.add_argument('--width', type=int, default=1920, help='输入图像宽度')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='使用 GPU 导出（可选）')

    args = parser.parse_args()
    export_onnx(args.output, args.height, args.width, args.use_gpu)
