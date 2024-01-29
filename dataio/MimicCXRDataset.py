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
from tqdm import tqdm


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


def _get_splits_MIMIC_CXR_frontal(cfg):
    view_pos = cfg.experiment.view_position
    if view_pos != 'FRONTAL':
        raise NotImplementedError("incorrect view position")

    # DATA LOADING
    print("loading PA")
    root_dir_pa = cfg.dataset.root_dir_PA
    img_mat_pa = np.load(os.path.join(root_dir_pa, "files_" + str(cfg.dataset.trans_resize) + ".npy"))
    df_pa = pd.read_csv(os.path.join(root_dir_pa, "meta_data.csv"))
    print("loading AP")
    root_dir_ap = cfg.dataset.root_dir_AP
    img_mat_ap = np.load(os.path.join(root_dir_ap, "files_" + str(cfg.dataset.trans_resize) + ".npy"))
    df_ap = pd.read_csv(os.path.join(root_dir_ap, "meta_data.csv"))

    # SUBJECT SPLIT
    # Load the original splits
    if cfg.experiment.splitting_method == 'original':
        split_df = pd.read_csv(os.path.join(cfg.dataset.root_dir_split, "mimic-cxr-2.0.0-split.csv"))
        train_subjects = split_df[split_df['split'] == 'train']['subject_id'].tolist()
        val_subjects = split_df[split_df['split'] == 'validate']['subject_id'].tolist()
        test_subjects = split_df[split_df['split'] == 'test']['subject_id'].tolist()

        subject_splits = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
    else:
        raise NotImplementedError('Only original split is supported at the moment')

    # DATA SPLIT
    out_dict = {}
    for split in subject_splits:
        images_list = []
        labels_list = []
        dicom_ids_list = []
        subject_ids_list = []
        for df, img_mat in zip([df_pa, df_ap], [img_mat_pa, img_mat_ap]):
            # label policy

            if cfg.experiment.label_policy == 'remove_uncertain':
                print("Dropping readings with uncertain values in selected class names")

                conditions = [(df[col] == -1) for col in cfg.experiment.target_list]
                combined_condition = pd.concat(conditions, axis=1).any(axis=1)
                df.drop(df[combined_condition].index, inplace=True)

                print('Number of filtered metadata rows: ', len(df))
            if cfg.experiment.label_policy == 'uncertain_to_negative':
                print("Setting uncertain values to negative in selected class names")
                for col in cfg.experiment.target_list:
                    df.loc[df[col] == -1] = 0


            # select relevant subjects
            # solo per debug TODO TOGLIERE
            if cfg.dataset.reduced_size:
                df = df.head(5000)
            df_split = df[df['subject_id'].isin(subject_splits[split])]
            df_split = df_split.sort_values(by=['subject_id']).reset_index()

            # extract dicom_id and labels and subject_id
            split_dicom = df_split['dicom_id']
            split_label = df_split[cfg.experiment.target_list]
            split_subject = df_split['subject_id']

            split_list = sorted(df.index[df['dicom_id'].isin(df_split['dicom_id'])].tolist())
            split_images = img_mat[split_list, :, :]

            print('Number of images in {} set: '.format(split), len(df_split))

            images_list.append(split_images)
            labels_list.append(split_label)
            dicom_ids_list.append(split_dicom)

        # create a single image matrix
        #image_mat = np.concatenate(images_list, axis=0)
        # create a single image matrix concat image_list in an efficient way
        image_mat = np.zeros(shape=(len(images_list[0]) + len(images_list[1]), images_list[0].shape[1], images_list[0].shape[2]))
        for i in tqdm(range(len(images_list[0]))):
            image_mat[i] = images_list[0][i]
            # delete the image from the list to save memory
            images_list[0][i] = None
        for i in tqdm(range(len(images_list[1]))):
            image_mat[i + len(images_list[0])] = images_list[1][i]
            # delete the image from the list to save memory
            images_list[1][i] = None



        # create a single label df
        label_df = pd.concat(labels_list, axis=0)

        # create a single dicom_id df
        dicom_ids_df = pd.concat(dicom_ids_list, axis=0)

        out_dict[split] = {'images': image_mat, 'labels': label_df, 'dicom_ids': dicom_ids_df, 'subject_ids': split_subject}

    return out_dict['train'], out_dict['val'], out_dict['test']


def get_splits_MIMIC_CXR(cfg):
    """Create train-validation-test split for the MIMIC-CXR dataset"""
    # TODO check if pl seed is enough TO REFACTOR
    np.random.seed(cfg.experiment.seed)

    # ASSERTIONS
    # assert cfg.datset is mimic cxr
    assert (cfg.dataset.name == 'MIMIC-CXR')

    if cfg.experiment.splitting_method != 'random' and cfg.experiment.splitting_method != 'original':
        raise NotImplementedError('Only random and original are supported at the moment')

    view_pos = cfg.experiment.view_position

    if view_pos != 'PA' and view_pos != 'AP' and view_pos != 'FRONTAL':
        raise NotImplementedError("view position not implemented")

    if view_pos == 'FRONTAL':
        return _get_splits_MIMIC_CXR_frontal(cfg)

    # MODALITY SELECTION
    # select the correct root_dir based on view position
    if view_pos == 'PA':
        root_dir = cfg.dataset.root_dir_PA
    if view_pos == 'AP':
        root_dir = cfg.dataset.root_dir_AP

    # load the correct img_mat
    if view_pos == 'PA' or view_pos == 'AP':
        img_mat = np.load(os.path.join(root_dir, "files_" + str(cfg.dataset.trans_resize) + ".npy"))
        df = pd.read_csv(os.path.join(root_dir, "meta_data.csv"))

    print('Root dir: ', root_dir)
    print('Image matrix shape: ', img_mat.shape)
    print('Number of metadata rows: ', len(df))

    # LABELING POLICY
    if cfg.experiment.label_policy == 'remove_uncertain':
        print("Dropping readings with uncertain values in selected class names")

        conditions = [(df[col] == -1) for col in cfg.experiment.target_list]
        combined_condition = pd.concat(conditions, axis=1).any(axis=1)
        df.drop(df[combined_condition].index, inplace=True)

        print('Number of filtered metadata rows: ', len(df))
    if cfg.experiment.label_policy == 'uncertain_to_negative':
        print("Setting uncertain values to negative in selected class names")
        for col in cfg.experiment.target_list:
            df.loc[df[col] == -1] = 0

    # SPLITTING METHOD
    # patient id split
    if cfg.experiment.splitting_method == 'random':
        patient_id = sorted(list(set(df['subject_id'])))

        train_idx, test_val_idx = train_test_split(patient_id, train_size=cfg.experiment.train_val_split,
                                                   shuffle=True, random_state=cfg.experiment.seed)

        test_idx, val_idx = train_test_split(test_val_idx, test_size=cfg.experiment.test_val_split,
                                             shuffle=True, random_state=cfg.experiment.seed)

        print('Number of patients in the training set: ', len(train_idx))
        print('Number of patients in the val set: ', len(val_idx))
        print('Number of patients in the test set: ', len(test_idx))

        splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    # CREATE OUT DICT
    out_dict = {}
    for split in splits:
        # select relevant subjects
        df_split = df[df['subject_id'].isin(splits[split])]
        df_split = df_split.sort_values(by=['subject_id']).reset_index()

        # extract dicom_id and labels
        split_dicom = df_split['dicom_id']
        split_label = df_split[cfg.experiment.target_list]

        split_list = sorted(df.index[df['dicom_id'].isin(df_split['dicom_id'])].tolist())
        split_images = img_mat[split_list, :, :]

        print('Number of images in {} set: '.format(split), len(df_split))
        out_dict[split] = {'ids': split_list, 'images': split_images, 'labels': split_label, 'dicom_ids': split_dicom}

    return out_dict['train'], out_dict['val'], out_dict['test']
