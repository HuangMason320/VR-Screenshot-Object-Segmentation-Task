#Set up
import numpy as np
import torch
# print(torch.backends.mps.is_available())
import matplotlib.pyplot as plt
import cv2

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
    
################################################################
# Set up for automatic mask generater
# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)
    
################################################################

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked at ({x}, {y})')
        input_points_list.append([x, y]) 
        input_label_list.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f'Clicked at ({x}, {y})')
        input_points_list.append([x, y]) 
        input_label_list.append(0)
    
image = cv2.imread('/Users/masonhuang/Pictures/Download/15257490252132.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

################################################################
# load the SAM model and predictor. Change the path below to point to the SAM checkpoint.
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "segment-anything-main/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("mps")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
################################################################
# produce an image embedding by calling SamPredictor.set_image. SamPredictor will use it for subsequent mask prediction
predictor.set_image(image)

################################################################
# choose a point that are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point)
input_points_list = []
input_label_list = []

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

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

################################################################
# multimask_output = true -> SAM return 3 outputs, = false -> return 1 output
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
################################################################
# (number_of_masks) x H x W -> C is the number of masks, and (H, W) is the original image size.
masks.shape  

# show the result of three outputs with Score
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()  


# input_point = np.array(input_points_list)
# input_label = np.array(input_label_list)

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
masks.shape

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
# show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show() 