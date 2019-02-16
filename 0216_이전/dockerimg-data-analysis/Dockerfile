# Use an official Python runtime as a parent image
FROM ubuntu:16.04

# Set the working directory to /app
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Update
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update


RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
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



RUN python3.6 -m pip install pip --upgrade


RUN python3.6 -m pip install -r ./requirements.txt
ENV LD_LIBRARY_PATH /usr/local/lib
# Make port 80 available to the world outside this container
#EXPOSE 80

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
#CMD ["python", "app.py"]


