# VR-Screenshot-Object-Segmentation-Task

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.  
  
Download the files  

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.  
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Model Checkpoints
Three model versions of the model are available with different backbone sizes. These models can be instantiated by running
```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```
Click the links below to download the checkpoint for vit_h.  
  
`vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

## Communicate b/w Unity & Python

https://github.com/Siliconifier/Python-Unity-Socket-Communication



