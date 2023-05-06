"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

from __future__ import print_function, division
from modules.utils.render import *
import sys, time
sys.path.append('..') # add parent directly for importing
import cv2
import argparse
import torch
import torch.nn.functional as F
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--datadir')
parser.add_argument('--format', default='avi', choices=['gif', 'avi'])

def create_gif_from_numpy_arrays(image_list, gif_filename, duration):
    """
    image_list: list of numpy arrays representing images
    gif_filename: str, output filename for the gif
    duration: float, duration between frames in seconds
    """
    # Normalize images and convert them to 8-bit format
    normalized_images = [((img - img.min()) * (255 / (img.max() - img.min()))).astype(np.uint8) for img in image_list]

    # Convert images to PIL.Image format
    pil_images = [imageio.core.util.Array(img) for img in normalized_images]

    # Save as GIF
    imageio.mimsave(gif_filename, pil_images, duration=duration)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill=' '):
    
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def generate_points_with_same_incident_angle(N):
    """
    Generates N points on the upper unit hemisphere with the same incident angle.
    """
    angles = np.linspace(0, 2 * np.pi, N)
    incident_angle = np.pi / 4  # Change this value to set a different incident angle

    x, y, z = np.cos(angles) * np.sin(incident_angle), np.sin(angles) * np.sin(incident_angle), np.cos(incident_angle) * np.ones(angles.shape)
    points = np.stack((x, y, z), axis=-1)

    return points

def numpy_to_pytorch(points):
    """
    Converts a NumPy array of points to a PyTorch Tensor.
    """
    return torch.tensor(points, dtype=torch.float32)

def create_video(images, output_file, fps=30, size=None):
    """
    Create a video from a list of NumPy images.

    :param images: A list of NumPy arrays (H, W, 3).
    :param output_file: Output video filename (e.g., 'output.avi').
    :param fps: Frames per second.
    :param size: Tuple (width, height). If None, size is obtained from the first image.
    """
    if size is None:
        height, width, _ = images[0].shape
        size = (width, height)

    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)

    for img in images:
        # Convert the image from RGB to BGR (as OpenCV uses BGR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    # Release the VideoWriter object and close the output file
    out.release()

def main():

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
    nml = (cv2.imread(f"{args.datadir}/normal.png")[:, :, ::-1]).astype(np.float32)/255.0
    base = (cv2.imread(f"{args.datadir}/baseColor.png")[:, :, ::-1]).astype(np.float32)/255.0
    rough = (cv2.imread(f"{args.datadir}/roughness.png")[:, :, ::-1]).astype(np.float32)/255.0
    metallic = (cv2.imread(f"{args.datadir}/metallic.png")[:, :, ::-1]).astype(np.float32)/255.0

    nml = 2 * torch.Tensor(nml).to(device).permute(2,0,1).unsqueeze(0) - 1
    base = torch.Tensor(base).to(device).permute(2,0,1).unsqueeze(0)
    rough = torch.Tensor(rough).to(device).permute(2,0,1).unsqueeze(0)[:,[0],:,:]
    metallic = torch.Tensor(metallic).to(device).permute(2,0,1).unsqueeze(0)[:,[0],:,:]


    height, width = nml.shape[-2:]
    if args.format == 'gif':
        max_size = 512
    if args.format == 'avi':
        max_size = 2048
    if np.max([height, width]) > max_size:
        aspect_ratio = width / height if width > height else height / width
        if height > width:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
        else:
            new_width = max_size
            new_height = int(max_size * aspect_ratio)


        nml = F.interpolate(nml, size=(new_width, new_height), mode='bilinear', align_corners=True)
        nml = F.normalize(nml, p=2, dim=1)
        base = torch.max(torch.min(torch.Tensor([1.0]).to(device), F.interpolate(base, size=(new_width, new_height), mode='bilinear', align_corners=True)),torch.Tensor([0.0]).to(device))
        rough = torch.max(torch.min(torch.Tensor([1.0]).to(device), F.interpolate(rough, size=(new_width, new_height), mode='bilinear', align_corners=True)),torch.Tensor([0.0]).to(device))
        metallic = torch.max(torch.min(torch.Tensor([1.0]).to(device), F.interpolate(metallic, size=(new_width, new_height), mode='bilinear', align_corners=True)),torch.Tensor([0.0]).to(device))
                                                                             
    N = 72  # Number of lighting
    points = generate_points_with_same_incident_angle(N)
    unit_vectors = numpy_to_pytorch(points)

    img_stack = []
    for k in range(unit_vectors.shape[0]):
        time.sleep(0.1)
        print_progress_bar(k+1, unit_vectors.shape[0], prefix='Render images', suffix='', length=30)
        l = torch.Tensor(unit_vectors[k,:]).reshape(1,3,1)
        nl, fd, fr = render(nml.to(device), l.to(device), base.to(device), rough.to(device), metallic.to(device), emit = 4.0, device = device)
        rendered = nl * (fd + fr) # (3, h, w)
        rendered = torch.clamp(rendered.cpu().permute(0,2,3,1).squeeze(), min=0, max=1).numpy() 
        img_stack.append((255.0 * rendered).astype(np.uint8))

    # Create a video from the list of images
    if args.format == 'avi':
        output_file = f'{args.datadir}/output.avi'
        create_video(img_stack, output_file)

    if args.format == 'gif':
        output_gif = f'{args.datadir}/output.gif'
        frame_duration = 0.05  # seconds
        create_gif_from_numpy_arrays(img_stack, output_gif, frame_duration)



if __name__ == '__main__':
    main()
