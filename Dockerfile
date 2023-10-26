FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
ENV POETRY_DOTENV_LOCATION=/root/.env
ENV PYTHONDONTWRITEBYTECODE 1
WORKDIR /master-thesis/
ADD . /master-thesis/

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get -y -qq install git gcc-c++ sudo --no-install-recommends --no-install-suggests && \
    sudo apt -y -qq install --reinstall software-properties-common --no-install-recommends --no-install-suggests && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt update --no-install-recommends --no-install-suggests && \
    sudo apt -y -qq install python3.11 python3.11-dev python3.11-distutils python3.11-venv --no-install-recommends --no-install-suggests && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1  && \
    python -m ensurepip && \
    python -m pip list -v && \
    python -m pip --version && \
    sudo update-alternatives --install /usr/bin/pip pip /usr/local/lib/python3.11/dist-packages/pip 1 && \ 
    sudo chmod 777 /usr/local/lib/python3.11/dist-packages/pip && \
    sudo apt -y -qq remove python3.10 python3.10-dev python3.10-distutils && \
    sudo apt -y autoremove && sudo apt -y clean

RUN --mount=type=cache,target=/root/.cache/pip/ \
    python -m pip install poetry --quiet && \
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.options.system-site-packages true && \
    poetry self add poetry-dotenv-plugin

RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install

RUN poetry run python --version && \
    poetry env info && \
    poetry run python -c "import diffccoder; import torch as t;" && \
    du -h --max-depth=1 ~/.cache/

RUN --mount=type=secret,id=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=MLFLOW_TRACKING_PASSWORD \
    --mount=type=secret,id=MLFLOW_TRACKING_URI \
    export MLFLOW_TRACKING_USERNAME=$(cat /run/secrets/MLFLOW_TRACKING_USERNAME) && \
    export MLFLOW_TRACKING_PASSWORD=$(cat /run/secrets/MLFLOW_TRACKING_PASSWORD) && \
    export MLFLOW_TRACKING_URI=$(cat /run/secrets/MLFLOW_TRACKING_URI) && \
    touch ~/.env && \
    sudo chmod 777 ~/.env && \
    ls -la ~/.env && \
    poetry run -vvv app set-dotenv MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI -vvv

ENV PYTHONDONTWRITEBYTECODE 0

ENTRYPOINT [ "poetry", "run", "app"]
