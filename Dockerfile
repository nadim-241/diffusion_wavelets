FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /usr/app
COPY src .
COPY requirements.txt .

ENV IMAGES_PATH=/usr/data/images
ENV WEIGHTS_PATH=/usr/data/weights

RUN pip install -r requirements.txt
CMD ["bash"]