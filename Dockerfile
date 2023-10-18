FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /master-thesis/
ADD . /master-thesis/

RUN apt-get update && \
    apt-get -y -qq install sudo --no-install-recommends --no-install-suggests

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN sudo apt -y -qq install --reinstall software-properties-common --no-install-recommends --no-install-suggests
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update --no-install-recommends --no-install-suggests && \
    apt -y -qq remove python3.10 python3.10-dev python3.10-distutils && \
    apt -y -qq install python3.11 python3.11-dev python3.11-distutils python3-pip --no-install-recommends --no-install-suggests

RUN pip install poetry --quiet

RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.options.system-site-packages true

RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install

ENTRYPOINT [ "poetry", "run", "app"]
