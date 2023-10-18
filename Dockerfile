FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED True
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /master-thesis/
ADD . /master-thesis/

#cuda compilation issues, it is needed to install python3-dev
RUN apt-get update && \
    apt-get -y -qq install sudo --no-install-recommends --no-install-suggests

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

RUN sudo apt -y -qq install --reinstall software-properties-common --no-install-recommends --no-install-suggests
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN sudo apt update && \
    apt -y -qq install python3-dev --no-install-recommends --no-install-suggests

RUN pip install poetry --quiet

RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.options.system-site-packages true

RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry install

ENTRYPOINT [ "poetry", "run", "app"]
