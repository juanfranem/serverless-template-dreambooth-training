import os
import json
import torch
import zipfile
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global vae
    model = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda")
    print("done")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: list) -> str:
    global model
    global vae

    # Setup concepts_list
    example_concepts_list = [
        {
            "instance_prompt": "photo of sks person",
            "class_prompt": "a photo of a person, ultra detailed",
            "instance_data_dir": "data/sks",
            "class_data_dir": "data/person"
        }
    ]


    # 'class_data_dir' contains regularization images
    # 'instance_data_dir' is where training images go
    for c in model_inputs:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    # Create concept file
    with open("concepts_list.json", "w") as f:
        json.dump(model_inputs, f, indent=4)

    # Call training script
    train = os.system("bash train.sh")
    print(train)

    # Compressed model to half size (4Gb -> 2Gb) to save space in gdrive folder: Models/
    steps = 1200
    compress = os.system(
        "python convert_diffusers_to_original_stable_diffusion.py --model_path 'stable_diffusion_weights/" + str(
            steps) + "/' --checkpoint_path /data/model/model.ckpt --half")
    print(compress)
    return str(train)
