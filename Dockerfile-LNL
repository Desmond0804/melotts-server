# Reference:
# https://pytorch.org/docs/main/notes/get_start_xpu.html
# https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-24-10
# https://dgpu-docs.intel.com/devices/hardware-table.html
# https://github.com/myshell-ai/MeloTTS/issues/257
# https://github.com/myshell-ai/MeloTTS/issues/126
# https://huggingface.co/mesolitica/MeloTTS-MS#requirements

FROM ubuntu:24.10
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Set build arguments for proxy
ARG http_proxy
ARG https_proxy
# Disable pip cache
ARG PIP_NO_CACHE_DIR=false

WORKDIR /melotts-server

COPY . .

# Update and install basic dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent gnupg wget software-properties-common git build-essential && \
    # 
    # Install compute-related & media-related packages
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libze-intel-gpu1 libze1 libze-dev intel-ocloc intel-opencl-icd clinfo \
    intel-gsc intel-metrics-discovery intel-media-va-driver-non-free libmfx1 \
    libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo ffmpeg && \
    # 
    # Install MeloTTS dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
    mecab libmecab-dev mecab-ipadic-utf8 espeak && \
    #
    # Install essential Python packages
    uv venv --python 3.11 && \
    uv pip install --upgrade pip setuptools wheel && \
    uv pip install -r /melotts-server/requirements-intel.txt --index-strategy unsafe-best-match && \
    uv run python -m unidic download && \
    # 
    # Clean up to reduce image size
    apt-get autoremove -y && \ 
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Download nltk_data
CMD uv run python -m nltk.downloader averaged_perceptron_tagger_eng && \
    #
    # Download TTS models and test run
    uv run init_downloads.py && \
    #
    # Start the MeloTTS server
    uv run app.py