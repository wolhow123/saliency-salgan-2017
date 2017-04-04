import os, sys

device = 'cpu' if int(os.environ["GPU"]) < 0 else 'cuda{}'.format(os.environ["GPU"])
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device={},floatX=float32,lib.cnmem=1.0,optimizer_including=cudnn,exception_verbosity=high".format(device)
sys.path.append(os.environ["SALGAN_PATH"] + "scripts/")

import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import riseml.server
import argparse
from PIL import Image
from io import BytesIO

model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)

load_weights(model.net['output'], path='gen_', epochtoload=90)

def predict(input_image):
   input_image = Image.open(BytesIO(input_image))

   img = np.array(input_image)

   size = (img.shape[1], img.shape[0])

   blur_size = 25

   if img.shape[:2] != (model.inputHeight, model.inputWidth):
     img = cv2.resize(img, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

   blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

   blob[0, ...] = (img.astype(theano.config.floatX).transpose(2, 0, 1))

   result = np.squeeze(model.predictFunction(blob))
   
   result = (result - np.min(result))/(np.max(result) -np.min(result))
   
   saliency_map = (result * 255).astype(np.uint8)

   # resize back to original size
   saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
   # blur
   saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
   # clip again
   saliency_map = np.clip(saliency_map, 0, 255)

   output_image = BytesIO()

   only_heatmap = int(os.environ["ONLY_HEATMAP"]) == 1

   if not only_heatmap:
      saliency_map = cv2.addWeighted(np.array(input_image), 0.3, cv2.cvtColor(cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.7, 0.0);
   
   med = Image.fromarray(saliency_map)
   med.save(output_image, format='JPEG')
      
   return output_image.getvalue()

riseml.server.serve(predict, port=os.environ.get('PORT'))