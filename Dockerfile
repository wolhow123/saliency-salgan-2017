FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN mkdir /app

#install cuda and cudnn
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 8.0.61
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-$CUDA_PKG_VERSION \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-8.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils\
            libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

#install anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

#install opencv
RUN conda install -y opencv

ENV PATH /opt/conda/lib/python2.7/site-packages:$PATH

#cloning repo
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