# -*- coding: utf-8 -*-
"""
Spyder Editor

This Script file is created to run on HPC- University of sydney

Add _main_ folder to the main directory

"""
##################################################################################################
def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
##########################################################
#custom code to predict selected classes

        # Create a list to store label and bounding box pairs
        label_bbox_pairs = []
        new_label = []
        new_bboxes = []
        new_segm_result = [] 
        
        # select only "person" labels
        for idx, label in enumerate(labels):
          # '0 is for "person" class'
          if label >= 62 and label <= 161:
            new_label.append(label)
            new_bboxes.append(bboxes[idx])
            # Append the label and its corresponding bounding box to the list
            label_bbox_pairs.append((label, bboxes[idx]))
            
            # Save the list of label and bounding box pairs to a file
            with open("label_bbox_pairs.txt", "w") as f:
                for pair in label_bbox_pairs:
                    f.write(f"Label: {pair[0]}, Bounding Box: {pair[1]}\n")
                
            if segm_result is not None and len(labels) > 0:
              new_segm_result.append(segms[idx])
        labels = np.array(new_label)
        bboxes = np.array(new_bboxes)
        if new_segm_result:
          segms = np.array(new_segm_result)
##################################################################
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
################################################################################################################

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
images = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000'  
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
image_input_dir = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000'

## Path to copy the test and train txt file
image_output_dir = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/data' 

## Split ratio
data_ratio = 0.05 
partition_seed = 2021


image_dir = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000' 
anno_dir = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Annotations_organized/first_1000/'  

## Creating pandas dataframe for dataset info
data_summ_df, object_dictionary = get_dataset_info(image_dir, anno_dir)

# Test & Train Partition into .txt file
image_partition(image_input_dir, image_output_dir, data_ratio, partition_seed)

# get label name list
class_list = get_label_list('/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/predefined_classes.txt')

print(len(object_dictionary))
print(len(class_list))

'''
# this code is raw and was created here and used for saving dataframe and object dictionary

import pickle
import pandas 

# to avoid parsing data and saving the dataframe to csv file and then later loading it 
data_summ_df.to_csv('/project/RDS-FEI-Age_saad_matloob-RW/Thesis_2022/HPC_codes/data_sum_df.csv')

#save object dictionary as saved_dictionary
with open('/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/saved_object_dictionary.pkl', 'wb') as f:
    pickle.dump(object_dictionary, f)
#%%
## Histogram Plot for the class

import matplotlib.pyplot as plt

plt.title('Class Distribution', {'fontsize': 25})
plt.rcParams["figure.figsize"] = (20,10)
plt.bar(object_dictionary.keys(), object_dictionary.values(), width = 1.0)
plt.xticks(rotation = 90)
y_seq = [0,200,400,600,800,1000,1200]
plt.yticks(y_seq)
plt.show

'''
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
    prefix = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Annotations_organized/first_1000/'
    
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
# Config File Setup
from mmdet.apis import set_random_seed
from mmcv import Config

cfg = Config.fromfile('/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/config_files/deformable_detr_twostage_refine_r50_16x2_50e_coco.py')

# Modify dataset type and path

cfg.dataset_type = 'NewDataset'
cfg.data_root = 'data/'

cfg.data.test.type = 'NewDataset'
cfg.data.test.data_root = 'data/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000/'

cfg.data.train.type = 'NewDataset'
cfg.data.train.data_root = 'data/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000/'

cfg.data.val.type = 'NewDataset'
cfg.data.val.data_root = 'data/'
cfg.data.val.ann_file = 'test.txt'
cfg.data.val.img_prefix = '/project/RDS-FEI-Age_saad_matloob-RW/Data/Jpeg_organized/first_1000/'

# modify num classes of the model in box head
cfg.model.bbox_head.num_classes = 162

#For the deformable DETR
model = dict(bbox_head=dict(as_two_stage=True))

# cfg.model.test_cfg.rcnn.score_thr = 0.5

# We can still use the pre-trained Mask RCNN model though we do not need to use the mask branch
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# original is the one on top and this needs to change if the training is stopped somwhere for some reason


cfg.load_from = '/project/RDS-FEI-Age_saad_matloob-RW/Data/HPC_codes/Model_checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72 .pth'
#cfg.load_from = '/media/saadi/Drive/Thesis_Folder/Thesis_2022/HPC_codes/model_checkpoints/epoch_14.pth'

# Set up working dir to save files and logs
cfg.work_dir = '/project/RDS-FEI-Age_saad_matloob-RW/Data/HPC_codes/checkpoints/'


cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 4

'''
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use 1 GPU
# Alternative: use 4 samples_per_gpu and lr = 0.02/4 below

# values below taken from the config file
cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 4


# Config file says that learning rate should not be disturbed
# cfg.optimizer.lr = 0.02
# cfg.lr_config.warmup = None

cfg.log_config.interval = 100

'''
# using the code below in the hashes from the config file and as a test
########################################################################
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
########################################################################

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 2
# We can set the checkpoint saving interval to reduce the storage cost usually 10
cfg.checkpoint_config.interval = 5
cfg.runner.max_epochs = 10


# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# cfg.model.backbone.norm_cfg.type = 'GN'
# cfg.model.backbone.norm_eval = False

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


#%%
## DETR-Deformable

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector

model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

#device error from some reason
cfg.device='cuda'

#train_detector(model, datasets, cfg, distributed=False, validate=True)

#%%

# Inference of Model 
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector

#img = '/content/drive/MyDrive/Colab Notebooks/Rad_files/first_1000/800.jpg'

#img = '/project/RDS-FEI-Age_saad_matloob-RW/Data/images_mae_final/ALGO_1/'

#Create s alist of file paths

import os
import glob

def list_image_files(root_dir, extensions=['.jpg', '.jpeg', '.png', '.gif']):
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.abspath(os.path.join(dirpath, filename)))
    return image_files

# Example usage:
root_directory = '/project/RDS-FEI-Age_saad_matloob-RW/Data/images_mae_final/main_images'
image_paths = list_image_files(root_directory)

# Print the list of image paths
print(image_paths)

#cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
#cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# we can not load from file because we have changed config file above
config_file = cfg

checkpoint_file = '/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/checkpoints_DETR_defor_HPC/epoch_50.pth''

#sftp://syou9359@hpc.sydney.edu.au/project/RDS-FEI-Age_saad_matloob-RW/HPC_codes/checkpoints_DDOD/epoch_250.pth

model = init_detector(config_file, checkpoint_file, device='cuda:0')

model = init_detector(config_file, checkpoint_file)
for path in image_paths:
    print(path)
    img = path
    result = inference_detector(model, img)
    # make the name dynamic
    output_dir = 'DETR_deform_output'
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    
    output_file = os.path.join(output_dir, os.path.basename(img))  # Use the input image file name
    
    model.show_result(img, result, score_thr=0.3, out_file = output_file)
    '''
    if result is not None:
        with open('items.txt', 'a') as file:
            file.write(img + "\n")
            for res in result:
                file.write(str(res) + "\n")
    else:
        print(f"Warning: Inference failed for image {img}. Skipping...")
        
    '''
    show_result_pyplot(model, img, result, score_thr=0.3)

# inference the demo image
#inference_detector(model, img)

#show_result_pyplot(model, img, result, score_thr=0.5)

#model.show_result(img, result, out_file='0617.jpg')# save image with result








