FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN mkdir /app

RUN apt-get update --fix-missing && apt-get install -y git wget 

#install anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

#install opencv
RUN conda install -y opencv

ENV PATH /opt/conda/lib/python2.7/site-packages:$PATH

#cloning salgan repo
RUN cd /app && git clone https://github.com/wolhow123/saliency-salgan-2017.git

#install lasagne
RUN pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt && \
	pip install https://github.com/Lasagne/Lasagne/archive/master.zip

#install pysaliency
RUN cd /app && git clone https://github.com/matthias-k/pysaliency.git && cd pysaliency/ && \
	python setup.py install

#install cmake
RUN apt-get install -y cmake

#install libgpuarray
RUN cd /app && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
	mkdir Build && cd Build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && make install &&\
	cd .. && python setup.py build && python setup.py install
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

#install riseml
RUN pip install riseml

#install requirements for salGAN
RUN cd /app/saliency-salgan-2017 && pip install -r requirements.txt
ENV SALGAN_PATH /app/saliency-salgan-2017/

#download weights
RUN wget --quiet https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz -P $SALGAN_PATH && \
	wget --quiet https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz -P $SALGAN_PATH && \
	wget --quiet https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl -P $SALGAN_PATH

CMD cd /app/saliency-salgan-2017/ && python salgan_demo.py