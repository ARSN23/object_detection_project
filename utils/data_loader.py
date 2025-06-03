import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as A
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def parse_voc_xml(xml_file, image_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    img_original = cv2.imread(img_path)
    h, w = img_original.shape[:2]
    img_resized = cv2.resize(img_original, (224, 224))

    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text.lower()
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) / w
        ymin = int(bbox.find('ymin').text) / h
        xmax = int(bbox.find('xmax').text) / w
        ymax = int(bbox.find('ymax').text) / h
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return img_resized, boxes, labels


def get_augmentor():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=30, p=0.5),
        A.MotionBlur(p=0.2),
        A.RandomGamma(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_sample(image, bbox, class_labels, augmentor):
    bbox = bbox[0]
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    try:
        aug = augmentor(image=image, bboxes=[[center_x, center_y, width, height]], class_labels=class_labels)
        b = aug['bboxes'][0]
        x1, y1, x2, y2 = b[0] - b[2]/2, b[1] - b[3]/2, b[0] + b[2]/2, b[1] + b[3]/2
        return cv2.resize(aug['image'], (224, 224)), [x1, y1, x2, y2], class_labels[0]
    except:
        return image, bbox, class_labels[0]


def load_dataset(xml_dir, image_dir, augment_data=False):
    images, bboxes, classes = [], [], []
    augmentor = get_augmentor()
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            img, box_list, label_list = parse_voc_xml(os.path.join(xml_dir, file), image_dir)
            for box, label in zip(box_list, label_list):
                images.append(img)
                bboxes.append(box)
                classes.append(label)
                if augment_data:
                    img_aug, box_aug, label_aug = augment_sample(img, [box], [label], augmentor)
                    images.append(img_aug)
                    bboxes.append(box_aug)
                    classes.append(label_aug)
    return np.array(images), np.array(bboxes), np.array(classes)


def prepare_labels(class_list):
    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(class_list)
    onehot = OneHotEncoder(sparse=False)
    onehot_encoded = onehot.fit_transform(integer_encoded.reshape(-1, 1))
    return onehot_encoded, encoder.classes_
