FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED True

WORKDIR /master-thesis/
ADD . /master-thesis/

#cuda compilation issues, it is needed to install python3-dev
RUN apt-get update && \
    apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN sudo apt -y install --reinstall software-properties-common
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update && \
    apt -y install python3.11 python3.11-dev python3.11-distutils
    
RUN pip install poetry
RUN poetry install

ENTRYPOINT [ "poetry", "run", "app"]
