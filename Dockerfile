FROM continuumio/anaconda:latest
MAINTAINER wesley goi <picy2k@gmail.com>

RUN git clone https://github.com/etheleon/vessel-scoring.git /tmp/vessel-scoring
WORKDIR /tmp/vessel-scoring
RUN python setup.py build && \
    python setup.py install && \
    pip install rolling-measures==0.0.5

RUN apt-get install libglu1 -y
