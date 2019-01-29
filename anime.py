import os

import cv2
import tqdm

from data import DataSet
from config import INPUT_DIR


def _is_image_file(file_name):
    return file_name.endswith('.jpg')


def read_data_set(size=(64, 64)):
    image_dir = os.path.join(INPUT_DIR, 'faces')

    print('Reading images...')

    images = []
    for file_name in tqdm.tqdm(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, file_name)
        if os.path.isfile(file_path) and _is_image_file(file_name):
            image = cv2.imread(file_path)
            image = cv2.resize(image, size)
            image = (image - 127.0) / 128.0
            images.append(image)

    print('Images are read successfully')

    data_set = DataSet(images)

    return data_set
