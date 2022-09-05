ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG GITHUB_USER
ARG GITHUB_TOKEN

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install nnfabrik

WORKDIR /src
RUN git clone -b transformer_readout https://github.com/KonstantinWilleke/neuralpredictors &&\
    git clone -b inception_loop https://github.com/sinzlab/mei

FROM ${BASE_IMAGE}
COPY --from=base /src /src
ADD . /src/nnvision

RUN python3.8 -m pip install --no-use-pep517 -e /src/neuralpredictors &&\
    python3.8 -m pip install --no-use-pep517 -e /src/nnvision &&\
    python3.8 -m pip install --no-use-pep517 -e /src/mei &&\
    python3.8 -m pip install git+https://github.com/sacadena/ptrnets &&\
    python3.8 -m pip install git+https://github.com/dicarlolab/CORnet
