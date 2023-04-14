# Imports
import pathlib

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize

from transformations import normalize_01, re_normalize
from unet import UNet
from PIL import Image

# def predict(img,
#             model,
#             preprocess,
#             postprocess,
#             device,
#             ):
#     model.eval()
#     img = preprocess(img)  # preprocess image
#     x = torch.from_numpy(img).to(device)  # to torch, send to device
#     with torch.no_grad():
#         out = model(x)  # send through model/network

#     out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
#     result = postprocess(out_softmax)  # postprocess outputs

#     return result

def predict(img,
            model,
            preprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()  # perform softmax on outputs

    return out_softmax

# root directory
root = pathlib.Path.cwd() / 'data' / 'Test'
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_names = get_filenames_of_path(root / 'Input')
targets_names = get_filenames_of_path(root / 'Target')

# read images and store them in memory
images = [imread(img_name) for img_name in images_names]
targets = [imread(tar_name) for tar_name in targets_names]

# Resize images and targets
images_res = [resize(img, (128, 128, 3)) for img in images]
# resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
# targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device =  torch.device('cpu')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


model_name = 'carvana_model.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name,map_location=torch.device('cpu'))

model.load_state_dict(model_weights)


# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img

# predict the segmentation maps 
# output = [predict(img, model, preprocess, postprocess, device) for img in images_res]

# Define the helper function
def decode_segmap(image, nc=2):
  label_colors = np.array([(0, 0, 0),(255, 255, 255)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)

  print(len(label_colors))
  return rgb

if __name__ == '__main__':
    
    for img,raw_img in zip(images_res,images):
        raw_img = Image.fromarray(raw_img, 'RGB').resize((1024,512))
        img = predict(img, model, preprocess, device)
        rgb = decode_segmap(img) 
        mask =  Image.fromarray(rgb, 'RGB').convert('L').resize((1024,512))
        mask.show()
        raw_img.show()


    # output = np.array([predict(img, model, preprocess, device) for img in images_res])

    # img = output[0]
    # print(img.shape)
    # print (np.unique(img))

    # from PIL import Image
    # rgb = decode_segmap(img) 
    # rgb =  Image.fromarray(rgb, 'RGB')
    # rgb.resize()
    # rgb.show()
