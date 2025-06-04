ARG DIST_ID
FROM nvidia/cuda:11.6.2-runtime-${DIST_ID}
ARG DEBIAN_FRONTEND=noninteractive
ADD . /code
WORKDIR /code
# Install Python and its tools
RUN apt update && apt install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    python3.9 \
    python3-pip \
    python3-setuptools \
    graphviz \
    xdg-utils

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN pip3.9 -q install pip --upgrade
RUN pip3.9 -q install wheel
RUN pip3.9 install -r requirements.txt