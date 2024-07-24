FROM tesis-predictive-model-training:latest


COPY ./requirements.txt .
#RUN ls /anaconda3 && exit 1
RUN /home/jupyter/anaconda3/bin/pip install -r ./requirements.txt

COPY /boot.sh .
ENTRYPOINT /bin/bash /boot.sh
CMD [ "/bin/bash" ]
