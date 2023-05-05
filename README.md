
## Scalable, Detailed, and Mask-Free Universal Photometric Stereo (CVPR2023 Hightlight)

<p align="center">
  <img src="./.images/teaser2.jpg" width="800px">
</p>

**Author:** Satoshi Ikehata (National Institute of Informatics https://satoshi-ikehata.github.io/)

**Conference:** IEEE/CVF Computer Vision and Pattern Recognition (CVPR), 2023

**Abstract:**
SDM-UniPS is Scalable, Mask-free, and Universal Photometric Stereo Networks. Our method can recover astonishingly intricate surface normal maps, rivaling the quality of 3-D scanners, even when images are captured under unknown, spatially-varying lighting conditions in uncontrolled environments. I provide not only a learning model for normal estimation but also for BRDF parameters (base color, roughness, metallic), as shown in Supplementary. Additionally, I offer code for rendering under a directional light source from both the normal and BRDF parameters.

## Example of Results
Given multiple images under different arbitrary lighting conditoins, you will get corresponding surface normal map and BRDF paramters as illustrated below:


## Required Dependencies
To successfully run the universal photometric stereo network, ensure that your system has the following dependencies installed:

- Python 3
- PyTorch
- OpenCV (cv2)
- Einops

## Tested Configurations
The network has been successfully tested on the following configurations:

- Ubuntu 20.04.5 and Windows 11, Python 3.11.3, PyTorch 2.0 with CUDA 11.8
- GPU: Nvidia RTX A6000 (48GB)

## Dataset Preparation
To run the universal photometric stereo network, you need shading images and an optional binary object mask. The object should be illuminated under arbitrary lighting sources, but shading variations should be sufficient (weak shading variations may result in poor results).

Organize your test data as follows (prefix):

```
YOUR_DATA_PATH
├── A (Suffix: default ".data")
│   ├── mask.png (optional)
│   ├── [Prefix (default: "0" (Train), "L" (Test))] imgfile1
│   ├── [Prefix (default: "0" (Train), "L" (Test))] imgfile2
│   └── ...
└── B (Suffix: default ".data")
    ├── mask.png (optional)
    ├── [Prefix (default: "0" (Train), "L" (Test))] imgfile1
    ├── [Prefix (default: "0" (Train), "L" (Test))] imgfile2
    └── ...
```

## Download Pretrained Models
Checkpoints of the network parameters for recovering both normal maps and BRDF maps are available [here](https://www.dropbox.com/s/yu8h6g0zp07mumd/checkpoint.zip?dl=0).

To use the pre-trained models, extract them as follows:

```
checkpoint
├── normal
│   └── nml.pytmodel
└── brdf
    └── brdf.pytmodel
```

## Running the Test
You can simply try the code using the datasets in my paper from [this link](https://www.dropbox.com/sh/afm4lkiz0iu1un3/AACfdKB66wl8iyFtJ4fzynGYa?dl=0).

To run the test, execute `main.py` with the following command:

```
python source/main.py --session_name YOUR_SESSION_NAME --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH
```

or

```
python source/main.py --session_name YOUR_SESSION_NAME --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH --scalable
```
'scalable' option is a powerful feature not included in the original framework. At the cost of some accuracy and computational expense, it enables normal and brdf estimation at any resolution with constant GPU memory consumption. For example, in the original implementation, inputting ten 2048x2048 resolution images consumes over 40GB of memory, but using this option reduces consumption to around 10GB. This consumption is proportional to the number of images but does not vary with resolution. The trick is very simple and I will add some details in the future. If you input images larger than 1024 x 1024, I highly recommend you to use this option.

If you don't need to recover either of normal map or brdf maps, please use specific options as

```
python source/main.py --session_name YOUR_SESSION_NAME --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH --target normal
```

or

```
python source/main.py --session_name YOUR_SESSION_NAME --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH --target brdf
```

By default, the maximum number of images to load is set to 10, and the maximum resolution is set to 4096x4096, regardless of the number of images or the resolution in the folder. If you want to change these numbers, please specify options as follow.

```
python source/main.py --session_name YOUR_SESSION_NAME --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH --max_image_res 10000 --max_image_num 100
```

The results will be saved in `YOUR_SESSION_NAME/results`. You will find a normal map and BRDF maps (base color, roughness, and metallic).

You can also use the provided code (`relighting.py`) for relighting the object under novel directional lights based on the recovered attributes. Follow the instructions displayed at the end of the prompt to use it. It should look like this.

```
 python source/relighting.py --datadir ./YOUR_SESSION_NAME/results/OBJECT_NAME.data
```

## Important Notice
This repository is for testing purposes only. I do not currently plan to release the training script and training data, unlike my previous projects. Please refrain from sending email requests for these resources.

## License

This project is licensed under the MIT License with a non-commercial clause. This means that you are free to use, copy, modify, and distribute the software, but you are not allowed to use it for commercial purposes. 

Copyright (c) [2023] [Satoshi Ikehata]

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{ikehata2023sdmunips,
  title={Scalable, Detailed and Mask-free Universal Photometric Stereo},
  author={Satoshi Ikehata},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

