FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get -y --fix-missing update
RUN apt-get -y install parallel

# Working directory
WORKDIR /home/app/gdn

# Set the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/home/app/gdn:/home/app/gdn/src:/home/app/gdn/src/baselines"

# Set wandb key (if you want to log to wandb)
ENV WANDB_API_KEY "type-key-here"

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3

# Install GDN and dependencies
COPY . /home/app/gdn
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Install environments
WORKDIR /home/app/gdn/src/envs/ic3net-envs
RUN python setup.py develop
WORKDIR /home/app/gdn
