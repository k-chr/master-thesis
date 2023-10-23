FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
ENV POETRY_DOTENV_LOCATION='~/.env'

WORKDIR /master-thesis/
ADD . /master-thesis/

RUN --mount=type=cache,target=~/.cache/pypoetry/cache \
    --mount=type=cache,target=~/.cache/pypoetry/artifacts \
    --mount=type=cache,target=~/.cache/pip/http \
    --mount=type=cache,target=~/.cache/pip/wheels \
    --mount=type=secret,id=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=MLFLOW_TRACKING_PASSWORD \
    --mount=type=secret,id=MLFLOW_TRACKING_URI \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get -y -qq install sudo --no-install-recommends --no-install-suggests && \
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
    sudo apt -y autoremove && sudo apt -y clean && \
    touch '/root/.env' && \
    sudo chmod 777 '/root/.env' && \
    python -m pip install poetry --quiet && \
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.options.system-site-packages true && \
    poetry self add poetry-dotenv-plugin && \
    poetry install && \
    poetry run python --version && \
    poetry env info && \
    poetry run python -c "import diffccoder; import torch as t; print(t.cuda.is_available())" && \
    poetry run app set-dotenv MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI

ENTRYPOINT [ "poetry", "run", "app"]
