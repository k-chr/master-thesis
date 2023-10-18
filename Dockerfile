FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED True

WORKDIR /master-thesis/
ADD . /master-thesis/

#cuda compilation issues, it is needed to install python3-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt-get update && \
    apt -y install python3.10-dev
RUN pip install poetry

RUN poetry env use system
RUN poetry config virtualenvs.options.system-site-packages false

RUN poetry install

ENTRYPOINT [ "poetry", "run", "app"]
