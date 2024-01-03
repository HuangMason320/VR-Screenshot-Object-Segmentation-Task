#Set up
import torch
import os
import numpy as np
import sys
sys.path.append("..")
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f'Clicked at ({x}, {y})')
        input_points_list.append([x, y]) 
        input_label_list.append(1)
    if event == cv2.EVENT_LBUTTONDOWN:
        input_box_list.append(x)
        input_box_list.append(y)
        print(f'BOX->Clicked at ({x}, {y})')

# Set up for automatic mask generater
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


input_path = ''
output_path = ''

image = cv2.imread(input_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load the SAM model, ONNX Model, and predictor.
sam_checkpoint = "segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
onnx_model_path = "../sam_onnx_quantized_example.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

device = torch.device("mps")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
# produce an image embedding by calling SamPredictor.set_image. SamPredictor will use it for subsequent mask prediction
predictor.set_image(image)

image_embedding = predictor.get_image_embedding().cpu().numpy()
image_embedding.shape

input_points_list = []
# labels 1 (foreground point) or 0 (background point)
input_label_list = []
input_box_list = []


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)

while True:
    # Display the image and wait for a key press
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        break
# Close the window
cv2.destroyAllWindows()

input_point = np.array(input_points_list)
input_label = np.array(input_label_list)
input_box = np.array(input_box_list)
# Fix Input
# input_point = np.array([[971, 1269]])
# input_label = np.array([1])
# input_box = np.array([[946, 1249], [997, 1301] ])

################################################################


onnx_box_coords = input_box.reshape(2, 2)
onnx_box_labels = np.array([2,3])

onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)
    
onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold

masks.shape

img_bgr = cv2.imread(input_path)
sam_mask_uint8 = (masks * 255).astype(np.uint8)

# Add alpha channel to original image
img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

# Scale mask to match image sizing
img_height, img_width = img_bgra.shape[0:2]
# resized_mask = cv2.resize(sam_mask_uint8, dsize = (img_width, img_height))

# Use the mask as the alpha channel of the image
img_bgra[:,:,3] = sam_mask_uint8

# Save png copy of masked image
cv2.imwrite("", img_bgra)



################################################################

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_mask(masks, plt.gca())
# if input_box.size != 0:
#     show_box(input_box, plt.gca())
# if input_point.size != 0:
#     show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.show() 

################################################################
# INPUT BOX
# if input_box.size != 0 and input_point.size == 0:
#     masks, _, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=input_box[None, :],
#         multimask_output=False,
#     )
#     masks.shape
# ################################################################

# if input_box.size != 0 and input_point.size != 0:
#     masks, _, _ = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         box=input_box,
#         multimask_output=False,
#     )

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
########################################################################
# SamAutomaticMaskGenerator implementation

# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)

# masks = sorted(masks, key=lambda x: x['area'], reverse=True)

# masks = sorted(masks, key=lambda x: x['area'], reverse=True)
# i = 0
# x, y, width, height = masks[0]['bbox']

# mask = masks[0]['segmentation']
# image[masks==False] = [255,255,255]

# cropped_image = image[int(y):int(y+height), int(x):int(x+width)]
# filename = os.path.join(output_path, 'Output' + '.png')
# cv2.imwrite(filename, image)

# for i in range(len(masks)):
#     image = cv2.imread(input_path)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
#     x, y, width, height = masks[i]['bbox']
    
#     cropped_image = image[int(y):int(y+height), int(x):int(x+width)]
    
#     mask = masks[i]['segmentation']
#     image[mask == False] = [255, 255, 255]

    
#     filename = os.path.join(output_path, str(i+50) + '.png')
#     cv2.imwrite(filename, cropped_image)