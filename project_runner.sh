#!/bin/bash

# Prompt for username and devices
read -p "Enter your Docker username: " USERNAME
read -p "Enter the GPU devices (e.g. 1): " DEVICES

# Define variables
IMAGE_NAME="$USERNAME/llm"
CONTAINER_NAME="restful_llama"
DOCKERFILE_PATH="."

# Build the Docker image
echo "Building Docker image..."
hare build -t $IMAGE_NAME $DOCKERFILE_PATH

# Run the Docker container
echo "Running Docker container..."
hare run -d --rm --gpus "\"device=$DEVICES\"" -v "$(pwd)":/code --name $CONTAINER_NAME $IMAGE_NAME

# Attach to the container and run the command
echo "Attaching to the Docker container and running the command..."
hare exec -it $CONTAINER_NAME poetry run python main.py

hare stop restful_llama
echo "Done!"

