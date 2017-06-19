FROM continuumio/anaconda:4.4.0
MAINTAINER wesley goi <picy2k@gmail.com>

RUN apt-get install gcc -y
RUN git clone https://github.com/DataKind-SG/vessel-scoring.git /tmp/vessel-scoring
WORKDIR /tmp/vessel-scoring
RUN python setup.py build && \
    python setup.py install && \
    pip install \
        rolling-measures==0.0.5 \
        gpsdio==0.0.7

RUN apt-get install libglu1 -y
