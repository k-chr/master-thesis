FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /master-thesis/
ADD . /master-thesis/

#cuda compilation issues, it is needed to install python3-dev
RUN apt-get update && \
    apt-get -y install sudo --no-install-recommends --no-install-suggests

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN sudo apt -y install --reinstall software-properties-common --no-install-recommends --no-install-suggests
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update && \
    apt -y install python3-dev --no-install-recommends --no-install-suggests

RUN pip install poetry

RUN poetry config virtualenvs.create false --local
RUN poetry config virtualenvs.options.system-site-packages true

RUN poetry install

ENTRYPOINT [ "poetry", "run", "app"]
