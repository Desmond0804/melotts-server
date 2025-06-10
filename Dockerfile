FROM ubuntu:22.04

# Set build arguments for proxy
ARG http_proxy
ARG https_proxy
# Disable pip cache
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1

WORKDIR /melotts-server

COPY . .

# Install dependencies and configure the environment
RUN set -eux && \
    # 
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 
    # Update and install basic dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
      wget git gnupg gpg-agent software-properties-common && \
    # 
    # Configure Intel OneAPI and GPU repositories
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    wget -O- https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    # 
    # Install compute-related packages
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libze-intel-gpu1 libze1 intel-opencl-icd clinfo libze-dev intel-ocloc espeak && \
    # 
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # 
    # Install essential Python packages
    pip install --upgrade pip setuptools wheel && \
    pip install -r /melotts-server/requirements-intel.txt && \
    python -m unidic download && \
    python -m nltk.downloader averaged_perceptron_tagger_eng && \
    # 
    # Clean up to reduce image size
    apt-get autoremove -y && \ 
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Download TTS models & Start the MeloTTS server
CMD python init_downloads.py && python app.py