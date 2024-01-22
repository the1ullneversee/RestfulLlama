ARG PYTHON_VERSION="3.10"
ARG BUILD_VARIANT="bullseye"
ARG RUNTIME_VARIANT="slim-bullseye"

FROM nvidia/cuda:11.0.3-base-ubuntu20.04
FROM python:${PYTHON_VERSION}-${BUILD_VARIANT}
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
# Install any python packages you need

WORKDIR /code
RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /code/
COPY . /code

RUN poetry config virtualenvs.in-project true

WORKDIR /code

RUN poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu111/torch_stable.html
RUN poetry add torch torchvision torchaudio
RUN poetry install
RUN poetry run pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2



RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so' >> ~/.bashrc 


RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN echo 'root:admin_tk_2024' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN apt install nano

# SSH login fix. Otherwise, the user is kicked off after login.
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

WORKDIR /code
