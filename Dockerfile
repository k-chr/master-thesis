FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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
    apt -y install python3.10-dev
RUN pip install poetry

RUN poetry env use system
RUN poetry config virtualenvs.options.system-site-packages false

RUN poetry install

ENTRYPOINT [ "poetry", "run", "app"]
