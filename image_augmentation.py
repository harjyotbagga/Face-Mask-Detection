import os
import numpy as np
from PIL import Image
from skimage import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

root_dir = 'OG Dataset'
without_mask_images = os.path.join(root_dir, 'without_mask')
with_mask_images = os.path.join(root_dir, 'with_mask')

dest_dir = os.path.join('dataset', 'aug')
without_mask_images_dest = os.path.join(dest_dir, 'without_mask')
with_mask_images_dest = os.path.join(dest_dir, 'with_mask')

datagen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
 )

for i, img_name in enumerate(os.listdir(with_mask_images)):
    # print(os.path.join(with_mask_images, img_name))
    img = io.imread(os.path.join(with_mask_images, img_name))
    io.imsave(os.path.join(with_mask_images_dest, img_name), img)
    img = np.expand_dims(img, 0)
    for aug_idx, batch in enumerate(datagen.flow(img, batch_size=1, save_to_dir=with_mask_images_dest, save_prefix='aug')):
        if (aug_idx > 5):
            break
    
for i, img_name in enumerate(os.listdir(without_mask_images)):
    # print(os.path.join(without_mask_images, img_name))
    img = io.imread(os.path.join(without_mask_images, img_name))
    io.imsave(os.path.join(without_mask_images_dest, img_name), img)
    img = np.expand_dims(img, 0)
    for aug_idx, batch in enumerate(datagen.flow(img, batch_size=1, save_to_dir=without_mask_images_dest, save_prefix='aug')):
        if (aug_idx > 5):
            break