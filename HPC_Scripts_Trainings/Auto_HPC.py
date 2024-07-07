# -*- coding: utf-8 -*-
"""
Spyder Editor

This Script file is created to run on HPC- University of sydney

Add _main_ folder to the main directory

"""

# importing modules
import os
import re
from shutil import rmtree
import shutil
import xml.etree.ElementTree as ET
import math
import random
import pandas as pd
from skimage import io

#%%
## Directory maker function

def dir_maker(data_dir = None, image_dir = None, anno_dir = None, checkpoints_dir = None, del_folder = True):

  loop_list = [data_dir, image_dir, anno_dir, checkpoints_dir]
  loop_final = []

  print('current working dir: ' + os.getcwd())

  for i in range(len(loop_list)):
    loop_test = loop_list[i]
    if loop_test != None:
      loop_final.append(loop_test)
      print('adding '+ loop_test +' folder to current working dir')
  
  if len(loop_final) < 0:
    print('no folder to be altered')
    print('exiting function')
  else:
    if os.getcwd() == '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes':
      for i in range(len(loop_final)):
        dir_test = loop_final[i]
        if os.path.exists(dir_test):
          if del_folder == True:
            rmtree(dir_test)
            print('deleting the existing "' + dir_test + '" folder')
            os.makedirs(dir_test)
            print('creating new "' + dir_test + '" folder')
          else:
            print('keeping the existing "' + dir_test + '" folder')
        else:
          os.makedirs(dir_test)
          print('creating new "' + dir_test + '" folder')
    else:
      print('Please set working directory to "/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes')

#%%

## Directory setup
data_dir = 'data'
image_dir = 'data/all_images'
anno_dir = 'data/all_annotations'
checkpoints = 'checkpoints'

dir_maker(data_dir, image_dir, anno_dir, checkpoints)

# Set the images and destination folders
images = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images'  
dest = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/data/all_images'

## Partition and iteration function to split between test, and train data
def image_partition(image_dir,output_dir=None,ratio=0.05, random_seed=None):

    """Check if 'test' and 'train' folder exist in destination
    
    Iterates through all .jpg and .xml file, randomly copy file into
    'test' and 'train' folder according to the split ratio
    
    Parameters:
    ----------
    source: str
        The path containing the label .jpg and .xml files
    
    dest: str
        The path to create new 'test' and 'train' folder
    
    ratio: float
        The split ratio between 'test and 'train' folder
    
    Returns:
    --------
    .jpg and .xml inside test folder
    .jpg and .xml inside train folder
    
    """

    if output_dir is None:
        output_dir = image_dir

    # images_raw = [f for f in os.listdir(image_dir)
              # if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

    if random_seed != None:
      random.seed(random_seed)
    
    images_raw = [f for f in os.listdir(image_dir)]

    images = []

    for i in range(len(images_raw)):
      x = os.path.splitext(images_raw[i])[0]
      images.append(x)

    num_images = len(images)
    print(num_images)
    
    num_test_images = math.ceil(ratio*num_images)
    test_images = []    

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        test_images.append(filename)
        images.remove(images[idx])
    
    with open('data/test.txt', 'w') as f:
          for item in test_images:
            f.write("%s\n" % item)

    with open('data/train.txt', 'w') as f:
          for item in images:
            f.write("%s\n" % item)

## Cross validation required later

#%%

def get_label_list(label_path):
    """Iterates through all label
    
    Parameters:
    ----------
    label_path: str
        The path containing the label .txt files
    
    Returns:
    --------
    Python list of the label.
    
    """
    
    label = pd.read_csv(label_path,header = None)
    label_list = []
    for i in range(len(label)):
        label_list.append(label[0][i])
        
    return label_list

def get_dataset_info(image_path, xml_path):
  
  dataset_list = []
  item_dict = {}

  xml_check = [f for f in os.listdir(xml_path) \
               if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.xml)$', f)]
  
  for xml_file in xml_check:
    xml_dir = os.path.join(xml_path + '/' + xml_file)
    print('Parsing file...........', xml_file)
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    filename = root.find('filename').text
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    img_dir = os.path.join(image_path + '/' + filename)
    img = io.imread(img_dir)
    if len(img.shape) > 2:
            bw_or_rgb = 'RGB'
    else:
        bw_or_rgb = 'BW'

    num_items = 0

    for member in root.findall('object'):
      bndbox = member.find('bndbox')
      value = (filename,
              width,
              height,
              member.find('name').text,
              ((int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)+1) * 
              (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)+1)),
              bw_or_rgb)
      dataset_list.append(value)
    
  column_name = ['filename', 'width', 'height',
                   'object_class', 'box size', 'channel']
  
  dataset_df = pd.DataFrame(dataset_list, columns = column_name)
    
  for i in range(len(dataset_df)):
    x = dataset_df.iloc[i][3] 
    if x not in item_dict:
      item_dict[x] = 1
    else:
      item_dict[x] += 1
  
  return dataset_df, item_dict

#%%

#uncomment/comment line 237 else copy paste the test and train files to data folder 
## Variable Setup

##Path to image file
image_input_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images/'

## Path to copy the test and train txt file
image_output_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/data' 

## Split ratio
data_ratio = 0.05 
partition_seed = 2021


image_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images' 
anno_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Annotations/'  

## Creating pandas dataframe for dataset info
data_summ_df, object_dictionary = get_dataset_info(image_dir, anno_dir)

# Test & Train Partition into .txt file
image_partition(image_input_dir, image_output_dir, data_ratio, partition_seed)

# get label name list
class_list = get_label_list('/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/predefined_classes.txt')

print(len(object_dictionary))
print(len(class_list))

#%%
## Data module load into MMDetection

import copy
import os.path as osp

import mmcv
import numpy as np
import xml.etree.ElementTree as ET

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class NewDataset(CustomDataset):
  
  CLASSES = class_list

  def load_annotations(self, ann_file):
    # Convert class label list into dict
    cat2label = {k: i for i, k in enumerate(self.CLASSES)}

    #load image name list from the .txt file (test/train)
    image_list = mmcv.list_from_file(self.ann_file)
    
    data_infos = []
    
    #convert annotations into middle format
    prefix = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Annotations/'
    
    for image_id in image_list:
      # opening each of the xml file from the all_annotations folder following
      # the test/train .txt list
      xml_file = f'{prefix}{image_id}.xml'

      # parse the xml using xml.etree.ElementTree
      tree = ET.parse(xml_file)
      root = tree.getroot()
      # Finding 'filename' xml tag
      filename = root.find('filename').text
      # Finding 'width' xml tag
      width = int(root.find('size').find('width').text)
      # Finding 'height' xml tag
      height = int(root.find('size').find('height').text)

      # Load all three info above into a dict
      data_info = dict(filename = f'{image_id}.jpg', width = width, height = height)

      # making empty list to hold the label & bbox info
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = [] # optional field

      # Finding all xml tag 'object'
      for member in root.findall('object'):
        # Finding xml tag 'bndbox' on each xml tag 'object'
        bndbox = member.find('bndbox')
        class_name = member.find('name').text
        # MMDetection bounding box format is [x1, y1, x2, y2]
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        
        # If we can find lable name in the cat2label add into gt_bboxes, else
        # put into gt_bboxes_ignore
        if class_name in cat2label:
          gt_labels.append(cat2label[class_name])
          gt_bboxes.append([x1, y1, x2, y2])
        else:
          gt_bboxes_ignore.append([x1, y1, x2, y2])
          gt_labels_ignore.append(-1) # Following MMDetection tutorial - not sure why this is -1

      # Making a dict from the list of info gathered from 1 image
      data_anno = dict(
          bboxes = np.array(gt_bboxes, dtype = np.float32).reshape(-1,4),
          labels = np.array(gt_labels, dtype = np.long),
          bboxes_ignore = np.array(gt_bboxes_ignore, dtype = np.float32).reshape(-1,4),
          labels_ignore = np.array(gt_labels_ignore, dtype = np.long))

      # Update the data_info dict with data_anno
      data_info.update(ann=data_anno)
      
      # Append the dict into data_infos list
      data_infos.append(data_info)

    return data_infos

#%%
#%%
# Config File Setup
from mmdet.apis import set_random_seed
from mmcv import Config

#
cfg = Config.fromfile('/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/config_files/autoassign_r50_fpn_8x2_1x_coco.py')
print('CFG file has been loaded')

# Modify dataset type and path

cfg.dataset_type = 'NewDataset'
cfg.data_root = 'data/'

cfg.data.test.type = 'NewDataset'
cfg.data.test.data_root = 'data/'
cfg.data.test.ann_file = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/data/train.txt'
cfg.data.test.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images/'

cfg.data.train.type = 'RepeatDataset'
cfg.data.train.type = 'NewDataset'
cfg.data.train.data_root = 'data/'
cfg.data.train.ann_file = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/data/train.txt'
cfg.data.train.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images/'

cfg.data.val.type = 'NewDataset'
cfg.data.val.data_root = 'data/'
cfg.data.val.ann_file = 'test.txt'
cfg.data.val.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/Final_images_annotations/Images/'

# modify num classes of the model in box head
cfg.model.bbox_head.num_classes = 162

# We can still use the pre-trained Mask RCNN model though we do not need to use the mask branch
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'


cfg.load_from = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/checkpoints_YOLOF/Model_checkpoints/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth'

# Set up working dir to save files and logs
cfg.work_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/checkpoints_Auto/'

#THE CODE BELOW IS TAKEN FROM https://ari08.medium.com/mmdetection-tutorial-in-kaggle-a-state-of-the-art-object-detection-library-b7c6d538f321

# The original learning rate (LR) is set for 8-GPU training.
# You divide it by 8 since you only use one GPU with Kaggle.

cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0.0))
    
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)



cfg.optimizer.lr = 0.001
cfg.lr_config.warmup_iters = 1000

runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, metric=['mAP'])


checkpoint_config = dict(interval=1)

cfg.checkpoint_config.interval = 5
cfg.total_epochs = 25
cfg.runner.max_epochs = 25
cfg.evaluation.metric = 'mAP'
cfg.evaluation.interval = 2


cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)



# Trying to remove init_cfg error
# Default = init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')

#cfg.init_cfg = None

################################################################################
# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')




from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector

model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

#print the working directory path
print('The working directory path is ',osp.abspath(cfg.work_dir))

#device error from some reason
cfg.device='cuda'

train_detector(model, datasets, cfg, distributed=False, validate=True)

#%%




