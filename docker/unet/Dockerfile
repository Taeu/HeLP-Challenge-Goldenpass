FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

WORKDIR /

COPY . .

RUN cd /usr/local/cuda/lib64 \ 
    && mv stubs/libcuda.so ./ \ 
    && ln -s libcuda.so libcuda.so.1 \ 
    && ldconfig

RUN apt-get update
RUN apt-get install -y autoconf \
											automake \
											cmake \
											git \
											libgtk2.0-dev \
											libjpeg-dev \
											liblcms2-dev \
											libpng-dev \
											libsqlite3-dev \
											libtiff-dev \
											libtool \
											libxml2-dev \
											pkg-config \
											python-pip \
											sqlite3 \
											wget

RUN git clone https://github.com/uclouvain/openjpeg.git 
RUN cd openjpeg && mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release && make && make install && make clean 

RUN git clone https://github.com/openslide/openslide.git
RUN cd openslide && autoreconf -i && ./configure && make && make install 

RUN wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
RUN tar xvzf openslide-3.4.1.tar.gz 
RUN cd openslide-3.4.1 && ./configure && make && make install && cd /

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
ENV LD_LIBRARY_PATH /usr/local/lib
