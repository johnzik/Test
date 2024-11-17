import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" #This disables oneDNN otherwise terminal sends messages about disabling oneDNN.
HF_HUB_OFFLINE=1  #So that it doesn't attempt to download from the HUB
HF_DATASETS_OFFLINE=1 #So that it only uses local datasets

from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline, FalconModel, FalconConfig 
import requests, torch
from PIL import Image
#from datasets import load_dataset

#Initialize the EdBianchi/ViT fire detect locally
LocalProcessor = AutoImageProcessor.from_pretrained("D:/Giannis/DUTH/UAV/AI/AI_Models/vit-fire-detection", local_files_only=True)
LocalModel = AutoModelForImageClassification.from_pretrained("D:/Giannis/DUTH/UAV/AI/AI_Models/vit-fire-detection", local_files_only=True)

#dataset = load_dataset("imagefolder", data_dir="C:/Giannis/DUTH/UAV/AI/ImageFolder")
#image = dataset["test"]["image"][0]

class fire_detect:
   def __init__(self):
        #Pipeline for image-classification using the local Processor and Model with PyTorch. 
        #Device= 0 is the CUDA id (cuda toolkit v12.4) and -1 is the CPU
        self.pipe = pipeline("image-classification", model=LocalModel, image_processor=LocalProcessor, framework="pt", device= 0)

   def detect(self, image_url): #Defines the detector func
        image = Image.open(requests.get(image_url, stream=True).raw) #Image obtained by cUrl request
        res = self.pipe(image)

        #inputs = processor(image, return_tensors="pt") #Image thrown into the processor with PyTorch
        #outputs = model(**inputs) #Outputs of the model in BatchFeature format (probably) - NOT JSON SERIALIZABLE
        #with torch.no_grad(): #no_grad for less memory consumption
        #   logits = model(**inputs).logits.tolist() #Convert logits to lists to make them JSON serializables

        return res

