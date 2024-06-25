FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
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

RUN poetry config virtualenvs.in-project false
RUN poetry env use /usr/bin/python3.10
RUN poetry run pip install anyio \
pytest \
scikit-build \
setuptools \
fastapi \
uvicorn \
sse-starlette \
pydantic-settings \
starlette-context \
huggingface-hub \
huggingface_hub[cli]
RUN poetry install
RUN poetry run pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN apt-get install build-essential g++ clang -y
RUN poetry run pip install --upgrade pip setuptools wheel
ENV CUDACXX="/usr/local/cuda-11.7/bin/nvcc"
RUN CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 poetry run pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
RUN echo 'export LLAMA_CPP_LIB=~/.cache/pypoetry/virtualenvs/fine-tune-MATOk_fk-py3.10/lib/python3.10/site-packages/llama_cpp/libllama.so' >> ~/.bashrc 
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
ENTRYPOINT ["poetry", "run", "python", "local_dp.py"]

# # Setup environment
# ENV TZ="UTC" \
#     MODEL_DOWNLOAD="True" \
#     MODEL_REPO="TheBloke/Llama-2-7B-Chat-GGUF" \
#     MODEL="llama-2-7b-chat.Q4_K_M.gguf" \
#     MODEL_PATH="/model" \
#     MODEL_ALIAS="llama-2-7b-chat" \
#     SEED=4294967295 \
#     N_CTX=1024 \
#     N_BATCH=512 \
#     N_GPU_LAYERS=0 \
#     MAIN_GPU=0 \
#     ROPE_FREQ_BASE=0.0 \
#     ROPE_FREQ_SCALE=0.0 \
#     MUL_MAT_Q=True \
#     LOGITS_ALL=True \
#     VOCAB_ONLY=False \
#     USE_MMAP=True \
#     USE_MLOCK=True \
#     EMBEDDING=True \
#     N_THREADS=4 \
#     LAST_N_TOKENS_SIZE=64 \
#     LORA_BASE="" \
#     LORA_PATH="" \
#     NUMA=False \
#     CHAT_FORMAT="llama-2" \
#     CACHE=False \
#     CACHE_TYPE="ram" \
#     CACHE_SIZE=2147483648 \
#     VERBOSE=True \
#     HOST="0.0.0.0" \
#     PORT=8000 \
#     INTERRUPT_REQUESTS=True \
#     QUIET="False"

# # Setup entrypoint
# COPY docker-entrypoint.sh /
# RUN chmod +x /docker-entrypoint.sh
# ENTRYPOINT ["/docker-entrypoint.sh"]

# HEALTHCHECK --interval=5s --timeout=10s --retries=3 \
#     CMD sv status /runit-services/llama-cpp-python || exit 1
