FROM nvidia/cuda:11.0.3-base-ubuntu20.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN apt-get install -y p7zip-full


WORKDIR /code
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /code/
COPY . /code
# unpack the files
RUN 7z x input_1.7z
RUN 7z x input_2.7z
RUN 7z x input_3.7z

RUN poetry config virtualenvs.in-project false
RUN poetry env use /usr/bin/python3.10
RUN poetry install
RUN poetry run pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
ENV PATH=/code/.venv/bin:$PATH

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