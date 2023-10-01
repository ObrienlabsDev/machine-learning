FROM tensorflow/tensorflow:latest-gpu
WORKDIR /src
COPY /src/tflow.py .
CMD ["python", "tflow.py"]

#RUN pip install -U jupyterlab pandas matplotlib
#EXPOSE 8888
#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
