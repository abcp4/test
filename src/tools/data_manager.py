import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def load_input(image_folder, data_desc, idx, rgb, beta, partition, normalize=False,add_pos=False):
    
    images = data_desc['image_names']
    indices = np.arange(len(images))
    
    if partition is not None:
        indices = np.array(partition)

    return load_image(image_folder + images[indices[idx]], rgb, beta, normalize=normalize,add_pos=add_pos )

def load_input_aug(data_desc, idx,partition=None,normalize=False,add_pos=False):
    
    aug_map = data_desc['aug_map']
    images = data_desc['image_names']
    indices = np.arange(len(images))
    
    if partition is not None:
        indices = np.array(partition)
        
    img_folder = data_desc['images_folder']
    augmented_folder = data_desc['augmented_images_folder']
    
    aug_images = []
    ref_image = images[indices[idx]]
    aug_images.append(load_image(img_folder + ref_image,normalize=normalize,add_pos=add_pos))
    for imagename in aug_map[ref_image]:
        aug_images.append(load_image(augmented_folder + imagename,normalize=normalize,add_pos=add_pos))
    return aug_images

def load_output(data_desc, idx, partition=None):
    
    classes = np.array(data_desc['image_classes'])
    indices = np.arange(len(classes))
    
    if partition is not None:
        indices = np.array(partition)
        
    return classes[indices[idx]]

def load_output_aug(data_desc, idx, partition=None):
    
    aug_map = data_desc['aug_map']
    classes = np.array(data_desc['image_classes'])
    images = data_desc['image_names']
    indices = np.arange(len(images))
    indices = np.arange(len(classes))
    
    if partition is not None:
        indices = np.array(partition)
        
    return [classes[indices[idx]]]*(1+len(aug_map[images[indices[idx]]]))

def load_sample(data_desc, idx, partition=None,normalize=False):
    return load_input(data_desc,idx,partition,normalize=normalize), load_output(data_desc,idx,partition)

def load_batch(image_folder, data_desc, batch_indices, rgb, partition=None,normalize=False, beta=99):
    batch_X = []
    batch_Y = []
    for i in tqdm(batch_indices):
        batch_X.append(load_input(image_folder, data_desc, i, rgb, beta, partition, normalize=normalize))
        batch_Y.append(load_output(data_desc, i, partition))
        
    return np.array(batch_X), np.array(batch_Y)

def load_batch_aug(data_desc, batch_indices, partition=None,normalize=False):
    batch_X = []
    batch_Y = []
    for i in batch_indices:
        batch_X.extend(load_input_aug(data_desc, i, partition,normalize=normalize))
        batch_Y.extend(load_output_aug(data_desc, i, partition))
        
    return np.array(batch_X), np.array(batch_Y)

def load_image(image_path, rgb, beta, add_pos=False, normalize=False ):
    image = cv2.imread(image_path)
    print(image_path)
    # changing image shape
    image = cv2.resize( np.array(image), (beta, beta), interpolation = cv2.INTER_AREA )

    # Convert to Pillow module
    #image = image.convert('RGB')
    image = Image.fromarray( image )

    # image to numpy array
    imdata = image.getdata()
    imsize = image.size
    vdata = np.array(imdata)
    # Corrigindo o tamanho para as novas imagens g2_...
    if not rgb: #len(vdata.shape) > 1:
        vdata = np.mean(vdata,axis=1)
    # sets all images to (-1,+1) range
    if normalize:
        vdata = 2 * (vdata / 255.0 - 0.5)
    # # generate positions feats
    if add_pos:
        dim = imsize[0]
        idx = np.arange(dim)
        idx_rel = idx / (dim - 1)
        pos_mat_arr = np.dstack(np.meshgrid(idx_rel, idx_rel)).reshape(dim * dim, 2)

        vdata = np.hstack((vdata, pos_mat_arr))
    if rgb:
        #vdata = vdata.reshape( (imsize[0], imsize[1], imsize[2]) )
        vdata = vdata.reshape((imsize[0], imsize[1], vdata.shape[1]))
        #vdata = vdata[:,:,0].reshape((imsize[0],imsize[1],1))
    else:
        vdata = vdata.reshape((imsize[0], imsize[1],1))
    # vdata = vdata.reshape((imsize[0], imsize[1]))
    return vdata

def split(data_desc, slice=1):
    data_desc['train'] = data_desc['train'][:int(len(data_desc['train'])*slice)]
    data_desc['dev'] = data_desc['dev'][:int(len( data_desc['dev']) * slice )]
    data_desc['test'] = data_desc['test'][:int(len( data_desc['test']) * slice )]
    return data_desc