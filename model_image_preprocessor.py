import tensorflow as tf


# IMG_SIZE = (224,224)
# BATCH_SIZE = 16

#
# Declaring global variables in python
# train_data = None
# test_data = None
# val_data = None
def preprocess_images_from_directory(train_dir_path, test_dir_path
                                     , label_mode, img_size, seed=45,
                                     batch_size=32, val_split=0.2, ):
    """
    :param seed:Optional random seed for shuffling and transformations.
    :param img_size: Size to resize images to after they are read from disk
    :param val_split:Optional float between 0 and 1, fraction of data to reserve for validation.
    :param train_dir_path: Directory path of the training dataset.
    :param test_dir_path: Directory path of the testing dataset.
    :param label_mode: The values can be int-> for `sparse_categorical_crossentropy` loss,
    categorical-> for categorical_crossentropy loss, binary -> binary_crossentropy
    :param (int) batch_size: Size of the batches of data. Default: 32.
    :return: (<class 'dict'>)
    {
        "train_data": train_data,
        "test_data": test_data,
        "val_data": val_data
    }
    """

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir_path,
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
        subset="training"  # must be used inconjuction with validation_split.
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir_path,
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
        subset="validation"  # must be used inconjuction with validation_split.
    )

    # 'If using `validation_split` and shuffling the data, you must provide '
    # 'a `seed` argument, to make sure that there is no overlap between the '
    # 'training and validation subset.'

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_path,
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False  # don't shuffle for prediction analysis
    )
    # global val_data used in conjuction with global variables declared outside the function

    # val_data = tf.keras.preprocessing.image_dataset_from_directory(
    #     val_dir_path,
    #     shuffle=True, #dont shuffle for prediction analysis
    #     label_mode=label_mode,
    #     image_size=(224,224),
    #     batch_size=BATCH_SIZE)

    return {
        "train_data": train_data,
        "test_data": test_data,
        "val_data": val_data
    }
