# SalGAN demo

1. Install nvidia docker. Make sure that the host has cuda driver installed, according to the [nvidia docker repo](https://github.com/NVIDIA/nvidia-docker)
    ```
    # Install nvidia-docker and nvidia-docker-plugin
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
    
    # Test nvidia-smi
    nvidia-docker run --rm nvidia/cuda nvidia-smi
    ```
2. Build the image (might be done with simple docker)
    ```
    cd saliency-salgan-2017/
    docker build -t "salgan_demo" .
    ```
3. Run the app
    ```
    # produce image+saliency map with 1st GPU
    nvidia-docker run -p 5000:5000 -e ONLY_HEATMAP=0 -e GPU=0 -it salgan_demo
    
    # produce only saliency map with CPU
    nvidia-docker run -p 5000:5000 -e ONLY_HEATMAP=1 -e GPU=-1 -it salgan_demo
    ```
4. Query
    ```
    curl -F image=@image.jpg http://localhost:5000/predict -o output.jpg
    ```

# SalGAN: Visual Saliency Prediction with Generative Adversarial Networks

| ![Junting Pan][JuntingPan-photo]  | ![Cristian Canton Ferrer][CristianCanton-photo]  |  ![Kevin McGuinness][KevinMcGuinness-photo] | ![Noel O'Connor][NoelOConnor-photo] | ![Jordi Torres][JordiTorres-photo] |![Elisa Sayrol][ElisaSayrol-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Junting Pan][JuntingPan-web]  | [Cristian Canton Ferrer][CristianCanton-web] | [Kevin McGuinness][KevinMcGuinness-web] | [Noel O'Connor][NoelOConnor-web] |  [Jordi Torres][JordiTorres-web] | [Elisa Sayrol][ElisaSayrol-web]  | [Xavier Giro-i-Nieto][XavierGiro-web]   |

[JuntingPan-web]: https://www.linkedin.com/in/junting-pan
[CristianCanton-web]: https://cristiancanton.github.io/
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[JordiTorres-web]: jorditorres.org
[ElisaSayrol-web]: https://imatge.upc.edu/web/people/elisa-sayrol
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[JuntingPan-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JuntingPan.jpg "Junting Pan"
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/Kevin160x160%202.jpg?token=AFOjyZmLlX3ZgpkNe60Vn3ruTsq01rD9ks5YdAaiwA%3D%3D "Kevin McGuinness"
[CristianCanton-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/CristianCanton.jpg?token=AFOjyS9qMOnUPVLZpqN80ChO0R-x0SI5ks5Yc3qJwA%3D%3D "Cristian Canton"
[JordiTorres-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/JordiTorres.jpg?token=AFOjyUaOhEyX2MGayU2C4tExpQeT0jFUks5Yc3vcwA%3D%3D
[ElisaSayrol-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/ElisaSayrol.jpg "Elisa Sayrol"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/NoelOConnor.jpg "Noel O'Connor"
[XavierGiro-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"

A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-microsoft] |![logo-bsc] | ![logo-gpi] |
|:-:|:-:|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web] | [Microsoft][microsoft-web]|[Barcelona Supercomputing Center][bsc-web] | [UPC Image Processing Group][gpi-web] |

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[microsoft-web]: https://www.microsoft.com/en-us/research/
[bsc-web]: https://www.bsc.es/
[upc-web]: http://www.upc.edu/?set_language=en
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-microsoft]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/logos/microsoft.jpg?token=AFOjyc8Q1kkjcWIP-yen0FTEo0lsWPk6ks5Yc3j4wA%3D%3D "Microsoft"
[logo-bsc]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/logos/bsc320x86.jpg?token=AFOjyWSHWWVvzTXnYh1DiFvH2VoWykA3ks5Yc6Q1wA%3D%3D
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"


## Abstract

We introduce SalGAN, a deep convolutional neural network for visual saliency prediction trained with adversarial examples.
The first stage of the network consists of a generator model whose weights are learned by back-propagation computed from a binary cross entropy (BCE) loss over downsampled versions of the saliency maps. The resulting prediction is processed by a discriminator network trained to solve a binary classification task between the saliency maps generated by the generative stage and the ground truth ones. Our experiments show how adversarial training allows reaching state-of-the-art performance across different metrics when combined with a widely-used loss function like BCE.

## Slides

<center>
<iframe src="//www.slideshare.net/slideshow/embed_code/key/5cXl80Fm2c3ksg" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/salgan-visual-saliency-prediction-with-generative-adversarial-networks" title="SalGAN: Visual Saliency Prediction with Generative Adversarial Networks" target="_blank">SalGAN: Visual Saliency Prediction with Generative Adversarial Networks</a> </strong> from <strong><a target="_blank" href="//www.slideshare.net/xavigiro">Xavier Giro</a></strong> </div>
</center>

## Publication

Find the pre-print version of our work on [arXiv](https://arxiv.org/abs/1701.01081).

![Image of the paper](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/figs/thumbnails.jpg)

Please cite with the following Bibtex code:

```
@InProceedings{Pan_2017_SalGAN,
author = {Pan, Junting and Canton, Cristian and McGuinness, Kevin and O'Connor, Noel E. and Torres, Jordi and Sayrol, Elisa and Giro-i-Nieto, Xavier and},
title = {SalGAN: Visual Saliency Prediction with Generative Adversarial Networks},
booktitle = {arXiv},
month = {January},
year = {2017}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Cristian Canton, Kevin McGuinness, Noel E. O'Connor, Jordi Torres, Elisa Sayrol and Xavier Giro-i-Nieto. "SalGAN: Visual Saliency Prediction with Generative Adversarial Networks." arXiv. 2017.*



## Models

The SalGAN presented in our work can be downloaded from the links provided below the figure:

SalGAN Architecture
![architecture-fig]

* [[SalGAN Generator Model (127 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz)
* [[SalGAN Discriminator (3.4 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz)

[architecture-fig]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/fullarchitecture.jpg?token=AFOjyaH8cuBFWpldWWzo_TKVB-zekfxrks5Yc4NQwA%3D%3D "SALGAN architecture"
[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt

## Visual Results

![Qualitative saliency predictions](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/qualitative.jpg?token=AFOjyaO0uT7l7qGzV7IyrcSgi8ieeayTks5Yc4s2wA%3D%3D)


## Datasets

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [SALICON](http://salicon.net/).

### Test
Two different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).


## Software frameworks

Our paper presents two convolutional neural networks, one correspends to the Generator (Saliency Prediction Network) and the another is the Discriminator for the adversarial training. To compute saliency maps only the Generator is needed.

### SalGAN on Lasagne

SalGAN is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
```
pip install -r https://github.com/imatge-upc/saliency-salgan-2017/blob/junting/requirements.txt
```

### Usage

To train our model from scrath you need to run the following command:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 02-train.py
```
In order to run the test script to predict saliency maps, you can run the following command after specifying the path to you images and the path to the output saliency maps:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 03-predict.py
```
With the provided model weights you should obtain the follwing result:

| ![Image Stimuli]  | ![Saliency Map]  |
|:-:|:-:|

[Image Stimuli]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/images/i112.jpg
[Saliency Map]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/saliency/i112.jpg

Download the pretrained VGG-16 weights from: [vgg16.pkl](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl)


## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the projects [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application) and [Malegra TEC2016-75976-R](https://imatge.upc.edu/web/projects/malegra-multimodal-signal-processing-and-machine-learning-graphs), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289. |  ![logo-ireland] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"
[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/saliency-salgan-2017/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.

<!---
Javascript code to enable Google Analytics
-->

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-7678045-13', 'auto');
  ga('send', 'pageview');

</script>
