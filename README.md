# VR-Screenshot-Object-Segmentation-Task

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.  
  
Download the files  

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.  
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
## Python for Unity
For Mac:  
1. Install [MacPorts](https://www.macports.org/) 
2. Install Python and PySide by pasting in the Terminal:  
```
sudo port install python27 py27-pyside
```
4. Within Unity, go to Edit -> Project Settings -> Python and set the out of process Python setting to read /opt/local/bin/python2.7  
5. Restart Unity.



