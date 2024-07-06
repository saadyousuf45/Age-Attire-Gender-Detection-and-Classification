<div align="center">
  <h1>Age-Gender-Attire in the Wild</h1>
</div>
<div align="center">
<img src="https://user-images.githubusercontent.com/33414652/48990447-91441200-f182-11e8-8ad7-d00d1e1f9147.jpg" alt="Description" width="70%">
</div>

## This Repo contains all the files necessary for the development and deployment of this Project.

This repository contains a link to a unique Dataset created to Simultaneously Detect Age, Gender and Attire in the wild.
It also contains many code-snippets I have written over the years to help with data-wrangling and combing. 
Please feel free to raise any issue and to reach out. 

## Table of Contents

  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Features

- Feature 1: Implementation of trained algorithms on 10 different object detectors.
- Feature 2: Dataset of images, with faces, identified with age and gender also accompanying all attire labels.
- Feature 3: SubDataset 


## Prerequisites
### Hardware:
GPU-enabled machine with at least 8Gb Graphics Card and 16 Gb of RAM.

### Software
Ubuntu or any other Linux-based system. 

#### Before installing PIP package or running any of the Python Jupiter Notebook codes 
#### Follow the following guideline to setup the environment to run my code.
This was taken from the following [link](https://github.com/open-mmlab/mmdetection/issues/10400)

#### Step 1 
conda create --name openmmlab python=3.8 -y Create conda env
#### Step 2 
conda activate openmmlab Activate conda env
#### Step 3 
conda install pytorch torchvision -c pytorch
#### Step 4 
pip install -U openmim
#### Step 5 
mim install mmengine
#### Step 6 
mim install "mmcv<=1.8.0" Install MMCV Older versions
#### Step 7 
git clone https://github.com/open-mmlab/mmdetection.git Cloning repo
#### Step 8 
cd mmdetection
#### Step 9 
git checkout 2.x Switching branches
#### Step 10 
pip install -v -e .
#### Step 11 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#### Step 12 
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
#### Step 13 
pip install future tensorboard
#### Step 14 
pip install setuptools==59.5.0
#### Step 15 
pip install notebook

Use cython==0.29.33 if your mmdet version requires mmpycocotools [[#10730.]](https://github.com/open-mmlab/mmdetection/issues/10730#issuecomment-1666391860)

I have tested it on mmdet=2.10 with pytorch=1.13 cuda=11.7.


## Installation
##### Pip python package based on the code.








# Contact
<p>
  
<a href="mailto:saadyousuf45@gmail.com">
  <img src="https://img.icons8.com/color/48/000000/gmail.png" alt="Gmail" width="30" height="30">
</a>
   _saadyousuf45@gmail.com_
<a href="https://www.linkedin.com/in/saad-b-yousuf-11640293/">
  <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn" width="30" height="30">
</a>
  _https://www.linkedin.com/in/saad-b-yousuf-11640293/_
</p>
