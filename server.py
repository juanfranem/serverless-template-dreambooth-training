# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field
import subprocess
import app as user_src

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
app = FastAPI(title="Stable Diffusion API")


class HealthCheckResponse(BaseModel):
    state: str = Field(..., title="State", description="State of system")
    gpu: bool = Field(..., title="GPU", description="State of gpu")


class TrainingRequest(BaseModel):
    unique_keys: set = Field(..., title="Training config", description="JSON training config")


class TrainingResult(BaseModel):
    log: str = Field(..., title="LogResult", description="Log of the result")
    time: float = Field(..., title="Time", description="Total duration of the training")


# Healthchecks verify that the environment is correct
@app.get('/healthcheck', response_model=HealthCheckResponse)
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True
    return HealthCheckResponse(state="state", gpu=gpu)


# Inference POST handler at '/' is called for every http call
@app.post('/', response_model=TrainingResult)
def inference(request: TrainingRequest):
    concept_list = []
    try:
        for key in request.unique_keys:
            key_dic = {
                'instance_prompt': f'photo of {key} person',
                'class_prompt': 'a photo of a person, ultra detailed',
                'instance_data_dir': f'/data/images/{key}',
                'class_data_dir': '/data/regularization/Mix'
            }
            concept_list.append(key_dic)
        output = user_src.inference(concept_list)
    except Exception as error:
        return HTTPException(500, detail=str(error))
    return TrainingResult(output)
