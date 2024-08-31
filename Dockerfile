FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git

RUN apt-get update && \
    apt-get install -y \
        python3-pip

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

WORKDIR /code
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /code/
COPY . /code
ENV CUDA_HOME=/usr/local/cuda-12.5
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64


RUN poetry config virtualenvs.in-project false
RUN poetry env use /usr/bin/python3.10
RUN poetry install

RUN poetry run pip install torch==2.4.0
RUN poetry run pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
RUN CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 poetry run pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade