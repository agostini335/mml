import importlib
import cs_manager

importlib.import_module('MyConfig')
from dataio.MimicCXRDataset import train_test_split_CXR
from torchvision import transforms


def prepare_dataset(cfg):
    # create datasets
    train_dict, test_dict, val_dict = train_test_split_CXR(cfg_dataset)
    transformList = [transforms.ToTensor()]

    transform = transforms.Compose(transformList)

    train_dataset = CXRDataset(label_name=cfg_dataset.class_names[0], labels_df=train_dict['labels'],
                               images=train_dict['images'], transform=transform)

    test_dataset = CXRDataset(label_name=cfg_dataset.class_names[0], labels_df=test_dict['labels'],
                              images=test_dict['images'], transform=transform)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)





@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyConfig):
    print(cfg)




if __name__ == "__main__":
    run_experiment()
