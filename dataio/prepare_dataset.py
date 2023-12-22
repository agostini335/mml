import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import zipfile
import hydra
import logging
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../configs', config_name='preprocessing_config')
def create_dataset(cfg: DictConfig):
    # logging initialization
    logging.basicConfig(filename="log/"+str(datetime.now())+cfg.logging.filename, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S', level=logging.INFO)

    # data paths initialization
    root_dir = cfg.root_dir
    res_dir = cfg.res_dir
    class_names = cfg.class_names
    data_dir = root_dir + 'files.zip'
    labels = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-chexpert.csv')

    # dropping readings with uncertain values in selected class names
    conditions = [(labels[col] == -1) for col in class_names]
    combined_condition = pd.concat(conditions, axis=1).any(axis=1)
    labels.drop(labels[combined_condition].index, inplace=True)

    # meta data loading
    df = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-metadata.csv')
    df = df[['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]

    # dropping rows with unwanted view position
    df = df[df['ViewPosition'] == cfg.view_position]

    # dropping duplicates for each patient
    df = df.drop_duplicates(subset=['subject_id'], keep='last')
    df = df.sort_values(by=['subject_id'])

    # merge with tables for patient and label information
    df = pd.merge(df, labels)

    # add a path column
    df['path'] = 'files/p' + df['subject_id'].astype(str).str[:2] + '/p' + df['subject_id'].astype(str) + '/s' + df[
        'study_id'].astype(str) + '/' + df['dicom_id'].astype(str) + '.jpg'

    # resize parameter
    transResize = cfg.trans_resize

    # img_mat initialization
    size = len(df)
    img_mat = np.zeros((size, transResize, transResize))
    df_filtered = df.copy()

    # transformation
    cnt = 0
    with zipfile.ZipFile(data_dir, 'r') as z:
        # iterate through files
        for filename in tqdm(df['path']):
            # read image
            try:
                img = Image.open(z.open(filename)).convert('RGB')
            except:
                # drop the row
                print(filename)
                df_filtered.drop(df_filtered[df_filtered['path'] == filename].index, inplace=True)
                # logging
                logging.info(f"dropped {filename}")
                continue
            
            # cut depending on the size
            width, height = img.size
            r_min = max(0, (height - width) / 2)
            r_max = min(height, (height + width) / 2)
            c_min = max(0, (width - height) / 2)
            c_max = min(width, (height + width) / 2)
            img = img.crop((c_min, r_min, c_max, r_max))

            # hist equalize and reshape
            img = img.resize((transResize, transResize))
            img = ImageOps.equalize(img)
            # TODO check if we can use RGB
            img = img.convert('L')

            # assign
            img_mat[cnt, :, :] = np.array(img)

            # increment
            cnt += 1
    
  
    # checking if the directory exist or not. 
    if not os.path.exists(res_dir): 
      
        # if the directory is not present then create it. 
        os.makedirs(res_dir) 

    # save
    img_mat = img_mat[0:cnt, :, :]
    np.save(res_dir + 'files_' + str(transResize) + '.npy', img_mat)

    # save dataframe as csv
    df_filtered.to_csv(res_dir + 'meta_data.csv', index=False)


def test_output():
    logging.basicConfig(filename="test.log", level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')
    logging.error("this is a test error")
    logging.info("this is a test info")


def main():
    create_dataset()


if __name__ == "__main__":
    main()
