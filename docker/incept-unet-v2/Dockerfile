FROM unet:0.0.1

WORKDIR /

COPY . .

RUN apt-get install -y libhdf5-serial-dev
RUN pip install -r ./requirements.txt
ENV LD_LIBRARY_PATH /usr/local/lib
