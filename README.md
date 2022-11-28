
# Dreambooth docker service

This repo provides a basic framework for training Dreambooth Stable Diffusion in production.

# Quickstart

docker build . \
    --build-arg HF_AUTH_TOKEN=[hugging_face_token]

docker run --gpus all \
    -v /data/model/:[path_where_save_model] \ 
    -v /data/images/:[path_where_we_have_images] \
    -p "80:8080"
