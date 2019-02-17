FROM excelsiorcjh/help-challenge:model

WORKDIR /

COPY . .

RUN apt-get install -y libhdf5-serial-dev
RUN pip install -r ./requirements.txt
ENV LD_LIBRARY_PATH /usr/local/lib
