import json
import os
from omegaconf import OmegaConf
from ultralytics.data.converter import convert_coco

PATH_MIX = './datasets/mix/mix_17'
os.system(f'mkdir -p {PATH_MIX}/annotations')

mot_json = json.load(open('/data/MOT17/annotations/train.json','r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = '/data/MOT17/train/' + img['file_name']
    img_list.append(img) 

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']


print('mot17')

max_img = 10000
max_ann = 2000000
max_video = 10

crowdhuman_json = json.load(open('/data/crowdhuman/annotations/train.json','r'))
img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/crowdhuman/Crowdhuman_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('crowdhuman_train')

video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_train'
})


max_img = 30000
max_ann = 10000000

crowdhuman_val_json = json.load(open('/data/crowdhuman/annotations/val.json','r'))
img_id_count = 0
for img in crowdhuman_val_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/crowdhuman/Crowdhuman_val/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_val_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('crowdhuman_val')

video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_val'
})

max_img = 40000
max_ann = 20000000

ethz_json = json.load(open('/data/ETHZ/annotations/train.json','r'))
img_id_count = 0
for img in ethz_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/ETHZ/' + img['file_name'][5:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in ethz_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('ETHZ')

video_list.append({
    'id': max_video,
    'file_name': 'ethz'
})

max_img = 50000
max_ann = 25000000

cp_json = json.load(open('/data/Citypersons/annotations/train.json','r'))
img_id_count = 0
for img in cp_json['images']:
    img_id_count += 1
    img['file_name'] = '/data/Citypersons/' + img['file_name'][12:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in cp_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('Cityscapes')

video_list.append({
    'id': max_video,
    'file_name': 'cityperson'
})

mot_json_train_for_val = json.load(open('/data/MOT17/annotations/train.json','r'))
img_list_val = []
for img in mot_json_train_for_val['images']:
    img['file_name'] = '/data/MOT17/train/' + img['file_name']
    img_list_val.append(img)
    
print('mot17_val_half')

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open(f'{PATH_MIX}/annotations/train.json','w'))
json.dump(mot_json_train_for_val, open(f'{PATH_MIX}/annotations/val.json','w'))

print('coco to yolo format')
convert_coco(labels_dir=f'{PATH_MIX}/annotations/', save_dir=f'{PATH_MIX}')

cfg = OmegaConf.create(dict(
    train=f'yolo_labels/images/train', 
    val=f'yolo_labels/images/val'
))
cls = OmegaConf.load('tools/coco_classes.yaml')
cfg = OmegaConf.merge(cfg, cls)
OmegaConf.save(config=cfg, f=f'{PATH_MIX}/train_config.yaml')