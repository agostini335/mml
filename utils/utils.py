from torchvision import transforms


def get_transformations(cfg):
    if cfg.dataset.augmentation == 'no':
        transform_list = [transforms.ToTensor()]
        train_transform = transforms.Compose(transform_list)
        test_transform = transforms.Compose(transform_list)
        val_transform = transforms.Compose(transform_list)
    elif cfg.dataset.augmentation == 'imagenet_style':
        transformList = [transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=(5)),
                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transform = transforms.Compose(transformList)
        transformList = [transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        test_transform = transforms.Compose(transformList)
        val_transform = transforms.Compose(transformList)
    else:
        raise NotImplementedError("transformations requested is not implemented")
    return train_transform, test_transform, val_transform
