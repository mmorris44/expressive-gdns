# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif
# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif
# Set flag for docker run command
BASE_FLAGS=-it --rm  -v ${PWD}:/home/app/gdn -w /home/app/gdn
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
IMAGE=wlgdn:latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

# Set example to run when using `make run`
# Default example
example=example.py

# Set wandb agent
# Default agent
agent=agent-name

.PHONY: wandb

# make file commands
run:
	$(DOCKER_RUN) python $(example)

bash:
	$(DOCKER_RUN) bash

wandb:
	$(DOCKER_RUN) wandb agent $(agent)

build:
	docker build --tag $(IMAGE) .