FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /master-thesis/
ADD . /master-thesis/

RUN apt-get update && \
    apt-get -y -qq install sudo --no-install-recommends --no-install-suggests

RUN useradd -m --no-log-init --system rekcod -g docker && echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER rekcod
RUN sudo apt -y -qq install --reinstall software-properties-common --no-install-recommends --no-install-suggests
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update --no-install-recommends --no-install-suggests && \
    apt -y -qq remove python3.10 python3.10-dev python3.10-distutils && \
    apt -y -qq install python3.11 python3.11-dev python3.11-distutils python3-pip --no-install-recommends --no-install-suggests
RUN alternatives --set python /usr/bin/python3.11 && alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install poetry --quiet

RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.options.system-site-packages true
RUN echo $(poetry config cache-dir)
RUN --mount=type=cache,target=~/.cache/pypoetry/cache \
    --mount=type=cache,target=~/.cache/pypoetry/artifacts \
    poetry install

RUN --mount=type=secret,id=MLFLOW_TRACKING_USERNAME \
    --mount=type=secret,id=MLFLOW_TRACKING_PASSWORD \
    --mount=type=secret,id=MLFLOW_TRACKING_URI \
    poetry run app set-dotenv MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_URI

ENTRYPOINT [ "poetry", "run", "app"]
