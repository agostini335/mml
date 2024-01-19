## MML

# steps to run experiments:

1. edit settings in configs folder:
   - preprocessing_config.yaml -> preprocessing settings
   - dataset_config.yaml -> dataset and dataloader settings
   - mml_experiments/MyConfig.py -> experiment settings

 2. run dataio/prepare_dataset.py -> obtain a numpy array of the dataset
 3. run notebooks/eda_dataset.py -> obtain statistics of the dataset
 4. run mml_experiments/main.py -> run the experiment