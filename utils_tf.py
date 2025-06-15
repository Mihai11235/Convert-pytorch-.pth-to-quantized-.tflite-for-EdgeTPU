from bdd_100k_dataset_local_tf import load_bdd100k_dataset_tf_1

def load_data_local_tf_1(batch_size=16, target_size=(128, 128)):
    train_ds = load_bdd100k_dataset_tf_1(
        images_dir='./copied/Dataset/100k_images_train/bdd100k/images/100k/train',
        masks_dir='./copied/Dataset/bdd100k_lane_labels_trainval/bdd100k/labels/lane/masks/train',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = load_bdd100k_dataset_tf_1(
        images_dir='./copied/Dataset/100k_images_val/bdd100k/images/100k/val',
        masks_dir='./copied/Dataset/bdd100k_lane_labels_trainval/bdd100k/labels/lane/masks/val',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = load_bdd100k_dataset_tf_1(
        images_dir='./copied/Dataset/100k_images_test/bdd100k/images/100k/test',
        masks_dir='./copied/Dataset/bdd100k_lane_labels_trainval/bdd100k/labels/lane/masks/test',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds, test_ds
