import cv2
from sklearn.preprocessing import MinMaxScaler
import patchify
import numpy as np
from PIL import Image

minmaxscaler = MinMaxScaler()
image_dataset = []
mask_dataset = []

def datset_load(dataset_root_folder,image_patch_size):
    for image_type in ['images' , 'masks']:
        if image_type == 'images':
            image_extension = 'jpg'
        elif image_type == 'masks':
            image_extension = 'png'
        for tile_id in range(1,8):
            for image_id in range(1,10):
                image = cv2.imread(f'{dataset_root_folder}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}',1)
                if image is not None:
                    if image_type == 'masks':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    size_x = (image.shape[1]//image_patch_size)*image_patch_size
                    size_y = (image.shape[0]//image_patch_size)*image_patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0,0, size_x, size_y))
                    image = np.array(image)
                    patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
            
                    for i in range(patched_images.shape[0]):
                        for j in range(patched_images.shape[1]):
                            if image_type == 'images':
                                individual_patched_image = patched_images[i,j,:,:]
                                individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                                individual_patched_image = individual_patched_image[0]
                                image_dataset.append(individual_patched_image)
                            elif image_type == 'masks':
                                individual_patched_mask = patched_images[i,j,:,:]
                                individual_patched_mask = individual_patched_mask[0]
                                mask_dataset.append(individual_patched_mask)
    
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)
    return image_dataset,mask_dataset