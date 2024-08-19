FROM continuumio/miniconda3

# TODO optimize. Reduce size
RUN apt update && apt install -y build-essential gfortran libatlas-base-dev python3-pip python3-dev \
    libopenblas-dev pkg-config libopenblas64-dev\
    && apt-get clean

WORKDIR /App
COPY ./PredictiveAPI/requirements.txt .
RUN pip install -r ./requirements.txt

VOLUME /Data
RUN mkdir -p /Data
RUN chown -R 1000:1000 /Data
ENTRYPOINT /bin/sh ./boot.sh
CMD [ "/bin/sh" ]

RUN mkdir /Lib
COPY ./PredictiveModel/Notebooks/*.py /Lib
COPY ./PredictiveAPI/App/ ./
COPY ./PredictiveAPI/boot.sh ./

