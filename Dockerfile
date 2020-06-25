FROM sinzlab/pytorch:latest

ADD . /src/nnfabrik
WORKDIR /src

RUN pip3 install -e nnfabrik
RUN pip3 install -e nnfabrik/ml-utils
RUN pip3 install -e nnfabrik/nnvision/nnvision
RUN pip3 install -e nnfabrik/mei/mei

WORKDIR /notebooks