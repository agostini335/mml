import os.path

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import pandas as pd
from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
from sklearn.model_selection import train_test_split

'''
class DatasetGenerator(Dataset):
    def __init__(self, imgs, img_list, label_list, transform):

        self.img_index = []
        self.listImageAttributes = []
        self.transform = transform
        self.imgs = imgs

        # Iterate over images and retrieve labels and protected attributes
        for i in range(len(img_list)):

            row = label_list.iloc[i]
            imageLabel = row['No Finding']
            imageAttr = np.array(row[row.index != 'No Finding'])
            imageAttr[np.isnan(imageAttr)] = 0

            if imageLabel == 1:
                imgLabel = 0
            else:
                imgLabel = 1

            self.img_index.append(imgLabel)
            self.listImageAttributes.append(imageAttr)

    def __getitem__(self, index):
        # Gets an element of the dataset

        imageData = Image.fromarray(self.imgs[index]).convert('RGB')

        image_label = self.img_index[index]

        image_attr = self.listImageAttributes[index]

        if self.transform != None: imageData = self.transform(imageData)

        # Return a tuple of images, labels, and protected attributes
        return {'img_code': index, 'labels': image_label,
                'features': imageData, 'concepts': image_attr}

    def __len__(self):

        return len(self.img_index)
'''


@hydra.main(version_base=None, config_path='../configs', config_name='dataset_config.yaml')
def train_test_split_CXR(cfg: DictConfig):
    """Performs train-validation-test split for the MIMIC-CXR dataset"""

    if cfg.splitting.method != 'random':
        raise NotImplementedError('Only random split is supported at the moment')

    # TODO set universal seed
    np.random.seed(cfg.seed)

    img_mat = np.load(os.path.join(cfg.root_dir, "files_" + str(cfg.trans_resize) + ".npy"))
    df = pd.read_csv(os.path.join(cfg.root_dir, "meta_data.csv"))

    print('Root dir: ', cfg.root_dir)
    print('Image matrix shape: ', img_mat.shape)
    print('Number of metadata rows: ', len(df))

    # patient id split
    patient_id = sorted(list(set(df['subject_id'])))

    train_idx, test_val_idx = train_test_split(patient_id, train_size=cfg.splitting.train_val_split,
                                               shuffle=True, random_state=cfg.seed)

    test_idx, val_idx = train_test_split(test_val_idx, test_size=cfg.splitting.test_val_split,
                                         shuffle=True, random_state=cfg.seed)

    print('Number of patients in the training set: ', len(train_idx))
    print('Number of patients in the val set: ', len(val_idx))
    print('Number of patients in the test set: ', len(test_idx))

    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    out_dict = {}

    for split in splits:
        df_split = df[df['subject_id'].isin(splits[split])]
        df_split = df_split.sort_values(by=['subject_id'])
        split_list = sorted(df.index[df['dicom_id'].isin(df_split['dicom_id'])].tolist())
        split_dicom = df_split['dicom_id']
        split_label = df_split[cfg.class_names]
        split_images = img_mat[split_list, :, :]
        print('Number of images in {} set: '.format(split), len(df_split))
        out_dict[split] = {'ids': split_list, 'images': split_images, 'labels': split_label, 'dicom_ids': split_dicom}

    return out_dict['train'], out_dict['val'], out_dict['test']


def get_CXR_dataloaders(dataset, root_dir, train_val_split=0.6, test_val_split=0.5, seed=42):
    """Returns a dictionary of data loaders for the MIMIC-CXR dataset, for the training, validation, and test sets."""

    train_list, val_list, test_list, train_label, val_label, test_label, \
        train_imgs, val_imgs, test_imgs = \
        train_test_split_CXR(dataset=dataset, root_dir=root_dir, train_val_split=train_val_split,
                             test_val_split=test_val_split, seed=seed)

    # Transformations
    transResize = 224
    transformList = []
    transformList.append(transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=(5)))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.Resize(size=transResize))
    transformList.append(transforms.ToTensor())
    transformList.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    train_transform = transforms.Compose(transformList)

    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.ToTensor())
    transformList.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transform = transforms.Compose(transformList)

    # Datasets
    image_datasets = {'train': ChestXRay_mimic_DatasetGenerator(imgs=train_imgs,
                                                                img_list=train_list,
                                                                label_list=train_label,
                                                                transform=train_transform),
                      'val': ChestXRay_mimic_DatasetGenerator(imgs=val_imgs,
                                                              img_list=val_list,
                                                              label_list=val_label,
                                                              transform=test_transform),
                      'test': ChestXRay_mimic_DatasetGenerator(imgs=test_imgs,
                                                               img_list=test_list,
                                                               label_list=test_label,
                                                               transform=test_transform)}

    return image_datasets['train'], image_datasets['val'], image_datasets['test']
