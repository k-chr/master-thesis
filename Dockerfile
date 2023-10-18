FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED True

WORKDIR /master-thesis/
ADD . /master-thesis/

#cuda compilation issues, it is needed to install python3-dev

RUN apt-get update && \
    apt-get install python3.11 && \
    apt-get -y install python3-dev

RUN pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install

ENTRYPOINT [ "poetry", "run", "app"]
