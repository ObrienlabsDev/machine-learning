#FROM tensorflow/tensorflow:latest-gpu
# https://hub.docker.com/layers/tensorflow/tensorflow/2.14.0-gpu/images/sha256-64602abcd8cc4f4bdd6268ca0abc39e6d37113d700886afd15f6dd151210b206?context=explore
FROM tensorflow/tensorflow:2.14.0-gpu
WORKDIR /src
COPY /src/tflow.py .
CMD ["python", "tflow.py"]

#RUN pip install -U jupyterlab pandas matplotlib
#EXPOSE 8888
#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
