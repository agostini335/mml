import os.path

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

class CXRDataset(Dataset):

    def __init__(self, label_name, labels_df, images, transform=None, target_transform=None):
        self.labels_df = labels_df
        self.label_name = label_name
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

        # create labels
        self.labels = self._create_labels()

    def _create_labels(self):
        # create labels based on a binary column
        labels = self.labels_df[self.label_name]
        labels = labels.fillna(0)
        # assert that values are either 0 or 1
        assert all((labels == 0) | (labels == 1))
        return labels

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):

        image = Image.fromarray(self.images[idx]).convert('RGB')

        label_values = self.labels.iloc[idx]

        # TODO CHECK THIS
        # If it's a single float64 value, convert it to a numpy array
        if isinstance(label_values, np.float64):
            label_values = np.array([label_values])

        # create a torch tensor from the numpy array
        label = torch.from_numpy(label_values.astype(int)).float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_splits_MIMIC_CXR(cfg):
    """Create train-validation-test split for the MIMIC-CXR dataset"""
    # TODO check if pl seed is enough
    np.random.seed(cfg.experiment.seed)

    # assert cfg.datset is mimic cxr
    assert (cfg.dataset.name == 'MIMIC-CXR')

    if cfg.experiment.splitting_method != 'random':
        raise NotImplementedError('Only random split is supported at the moment')

    view_pos = cfg.experiment.view_position

    if view_pos != 'PA' and view_pos != 'AP':
        raise NotImplementedError("view position not implemented")

    # select the correct root_dir based on view position
    if view_pos == 'PA':
        root_dir = cfg.dataset.root_dir_PA
    if view_pos == 'AP':
        root_dir = cfg.dataset.root_dir_AP

    # load the correct img_mat
    if view_pos == 'PA' or view_pos == 'AP':
        img_mat = np.load(os.path.join(root_dir, "files_" + str(cfg.dataset.trans_resize) + ".npy"))
        df = pd.read_csv(os.path.join(root_dir, "meta_data.csv"))
    else:
        raise NotImplementedError("not implemented")

    print('Root dir: ', root_dir)
    print('Image matrix shape: ', img_mat.shape)
    print('Number of metadata rows: ', len(df))

    if cfg.experiment.label_policy == 'remove_uncertain':
        print("Dropping readings with uncertain values in selected class names")

        conditions = [(df[col] == -1) for col in cfg.experiment.target_list]
        combined_condition = pd.concat(conditions, axis=1).any(axis=1)
        df.drop(df[combined_condition].index, inplace=True)

        print('Number of filtered metadata rows: ', len(df))

    # patient id split
    patient_id = sorted(list(set(df['subject_id'])))

    train_idx, test_val_idx = train_test_split(patient_id, train_size=cfg.experiment.train_val_split,
                                               shuffle=True, random_state=cfg.experiment.seed)

    test_idx, val_idx = train_test_split(test_val_idx, test_size=cfg.experiment.test_val_split,
                                         shuffle=True, random_state=cfg.experiment.seed)

    print('Number of patients in the training set: ', len(train_idx))
    print('Number of patients in the val set: ', len(val_idx))
    print('Number of patients in the test set: ', len(test_idx))

    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    out_dict = {}

    for split in splits:
        df_split = df[df['subject_id'].isin(splits[split])]
        df_split = df_split.sort_values(by=['subject_id']).reset_index()

        split_dicom = df_split['dicom_id']
        split_label = df_split[cfg.experiment.target_list]

        split_list = sorted(df.index[df['dicom_id'].isin(df_split['dicom_id'])].tolist())
        split_images = img_mat[split_list, :, :]

        print('Number of images in {} set: '.format(split), len(df_split))
        out_dict[split] = {'ids': split_list, 'images': split_images, 'labels': split_label, 'dicom_ids': split_dicom}

    return out_dict['train'], out_dict['val'], out_dict['test']




'''
todo refactoring
TO REMOVE


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

    if cfg.uncertain_values == 'exclude':
        print("Dropping readings with uncertain values in selected class names")

        conditions = [(df[col] == -1) for col in cfg.class_names]
        combined_condition = pd.concat(conditions, axis=1).any(axis=1)
        df.drop(df[combined_condition].index, inplace=True)

        print('Number of filtered metadata rows: ', len(df))

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
        df_split = df_split.sort_values(by=['subject_id']).reset_index()

        split_dicom = df_split['dicom_id']
        split_label = df_split[cfg.class_names]

        split_list = sorted(df.index[df['dicom_id'].isin(df_split['dicom_id'])].tolist())
        split_images = img_mat[split_list, :, :]

        print('Number of images in {} set: '.format(split), len(df_split))
        out_dict[split] = {'ids': split_list, 'images': split_images, 'labels': split_label, 'dicom_ids': split_dicom}

    return out_dict['train'], out_dict['val'], out_dict['test']

'''
