FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
ENV POETRY_DOTENV_LOCATION='~/.env'
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /master-thesis/
ADD . /master-thesis/

RUN apt-get update && \
    apt-get -y -qq install sudo --no-install-recommends --no-install-suggests

RUN groupadd -f docker && \
    useradd -ms /bin/bash --no-log-init --system rekcod -g docker && \
    usermod -aG sudo rekcod && \
    echo "rekcod ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER rekcod

RUN sudo apt -y -qq install --reinstall software-properties-common --no-install-recommends --no-install-suggests
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update --no-install-recommends --no-install-suggests && \
    apt -y -qq remove python3.10 python3.10-dev python3.10-distutils && \
    apt -y -qq install python3.11 python3.11-dev python3.11-distutils python3-pip --no-install-recommends --no-install-suggests
RUN alternatives --set python /usr/bin/python3.11 && alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install poetry --quiet && \
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.options.system-site-packages true && \
    poetry self add poetry-dotenv-plugin

RUN --mount=type=cache,target=~/.cache/pypoetry/cache \
    --mount=type=cache,target=~/.cache/pypoetry/artifacts \
    poetry install

RUN --mount=type=secret,id=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=MLFLOW_TRACKING_PASSWORD \
    --mount=type=secret,id=MLFLOW_TRACKING_URI \
    poetry run app set-dotenv MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI

RUN sudo apt autoremove && sudo apt clean

ENTRYPOINT [ "poetry", "run", "app"]
