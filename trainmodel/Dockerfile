FROM drtools/dask:latest
COPY . /opt/trainmodel

USER root

RUN pip install -r /opt/trainmodel/requirements.txt

WORKDIR /opt/trainmodel

