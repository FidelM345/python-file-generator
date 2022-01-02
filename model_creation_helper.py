import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras import layers
import numpy as np


# preprocess your images from the train, test, and validation directories into a format tha can
# be used by the tensorflow deep learning models.
def preprocess_images_from_directory_val_separate(train_dir_path, test_dir_path,
                                                  val_dir_path,
                                                  label_mode, img_size, seed=45,
                                                  batch_size=32, val_split=0.2, ):
    """
    :param seed:Optional random seed for shuffling and transformations.
    :param img_size: Size to resize images to after they are read from disk
    :param val_split:Optional float between 0 and 1, fraction of data to reserve for validation.
    :param train_dir_path: Directory path of the training dataset.
    :param test_dir_path: Directory path of the testing dataset.
    :param val_dir_path: Directory path for the validation dataset.
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
        # validation_split=val_split,
        seed=seed,
        # subset="training"  # must be used inconjuction with validation_split.
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir_path,
        label_mode=label_mode,
        image_size=img_size,
        batch_size=batch_size,
        # validation_split=val_split,
        seed=seed,
        # subset="validation"  # must be used inconjuction with validation_split.
    )

    # 'If using `validation_split` and shuffling the data, you must provide '
    # 'a `seed` argument, to make sure that there is no overlap between the '
    # 'training and validation subset.'

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_path,
        label_mode=label_mode,
        seed=seed,
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
        "val_data": val_data,
        "test_data": test_data
    }


# this function will return the number of files stored in the given directory
def check_no_files_in_folder(folder_path):
    # import os

    APP_FOLDER = folder_path

    totalFiles = 0
    totalDir = 0

    for base, dirs, files in os.walk(APP_FOLDER):
        print('Searching in : ', base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1

    print('Total number of files', totalFiles)
    print('Total Number of directories', totalDir)
    print('Total:', (totalDir + totalFiles))


# Generate more image files using data augmentation
def generate_augment_images(
        img_src_folder_path,
        img_destination_folder_path,
        no_of_iterations,
        batch_size=32,
        save_prefix="aug",
        save_format="png",
        datagen=ImageDataGenerator(
            rotation_range=40)

):
    # from keras.preprocessing.image import ImageDataGenerator
    # import os
    """
    :param (String) img_src_folder_path:  directory path to the src folder containing the images
    :param (String) img_destination_folder_path: directory path to the destination folder where augmented images will
    bes stored
    :param (int) no_of_iterations: the number of times the image batches can generate the required augmented image files e.g
    to generate 1000 images for batch size of 32 will be 1000/32 = 31 iterations.
    :param (int) batch_size:
    :param (String) save_prefix: the prefix in which your images will be saved into.
    :param (String) save_format: the format in which the images will be saved e.g png, jpeg,gif etc..
    :param datagen: the ImageDataGenerator object
    :return:
    """

    i = 0
    for batch in datagen.flow_from_directory(
            img_src_folder_path,
            batch_size=batch_size,
            save_to_dir=img_destination_folder_path,
            save_prefix=save_prefix,
            save_format=save_format):

        i += 1
        if i > no_of_iterations:
            break

    APP_FOLDER = img_destination_folder_path

    totalFiles = 0
    totalDir = 0

    for base, dirs, files in os.walk(APP_FOLDER):
        print('Searching in : ', base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1

    print('Total number of files', totalFiles)
    print('Total Number of directories', totalDir)
    print('Total:', (totalDir + totalFiles))


# Used to plot and visualize images using matplot library.
def visualize_and_show_images(
        train_data,
        class_names  # prepocessed_image_dictionary["train_data"].class_names
):
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import os
    # import tensorflow as tf

    # train_data = prepocessed_image_dictionary["train_data"]
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            print(tf.cast(labels[i], tf.int32).numpy()[0])
            plt.title(class_names[tf.cast(labels[i], tf.int32).numpy()[0]])
            plt.axis("off")


# create checkpoint callback
def saved_model_weights(saved_model_weight_name):
    """
    @param (String): saved_model_weight_name: supply the name in which the model weights
    will be saved in.
    """
    check_point_path = saved_model_weight_name + "_weights/cp.ckpt"
    check_point_callback = tf.keras.callbacks.ModelCheckpoint(
        check_point_path,
        save_weights_only=True,
        monitor="val_accuracy",
        save_best_only=True  # will only save the weights with highest accuracy
    )
    return check_point_callback


# Early stopping callback.
def early_stopping_callback(patience=7, restore_best_weights=True, monitor="val_accuracy"):
    """
  @params (int) patience: number of epochs to tolerate before stopping trainnig.
  @params (bolean) restore_best_weights: will restore the best learned model
  weights.
  @params (String) monitor: which model metric to monitor can either be `loss` or
  `val_accuracy`.
  """
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=restore_best_weights,
        monitor=monitor
    )
    return early_stopping_callback


def create_vanilla_model(
        train_data,
        val_data,
        callbacks,
        base_model, output_layer_filter_no=1,
        dropout_value=0.4,
        epochs=10,
        preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input,
        output_activation="sigmoid",
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        img_input_shape=(224, 224, 3)
):
    # base_model = tf.keras.applications.MobileNetV2(include_top=False,
    #                                                   weights="imagenet",
    #                                                   input_shape=img_input_shape)

    base_model.trainable = False

    inputs = layers.Input(shape=img_input_shape, name="input_layer")

    # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    x = preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    x = layers.Dropout(dropout_value)(x)
    outputs = layers.Dense(output_layer_filter_no, activation=output_activation, name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile and fit the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    training_history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        callbacks=callbacks
    )

    return {
        "model": model,
        "base": base_model,
        "training_history": training_history
    }


# Categorizes your folder containing the image files into test, train and validation sets depending on the arguments,
# provide to the function.
def train_test_split(source_dir_path, destination_dir_path, class_names, val_ratio, test_ratio,
                     same_file_number=False, no_files_in_folder=1345):
    """
    Categorizes your folder containing the image files into test, train and validation sets depending on the arguments,
    provide to the function.
    :param (int) no_files_in_folder: this value will be used if the value of same_file_number is set to True.
    :param (boolean) same_file_number: Determines whether a folder or classes should contain the same number of files or
    not. The  default value is False.
    :param (str) source_dir_path: the path of the folder containing the images
    :param (str) destination_dir_path: the path of the
    newly created folder that will contain the categorized image files - training, validation and test folders.
    :param (list) class_names: these are the names of the types of images we are going to train our model with.
    :param (float) val_ratio: this is the proportion or percentage of our data that will be validation data
    :param (float) test_ratio: this is the percentage or proportion of our data that will be considered as the test data.
    :return: (void) it has no return value.
    """
    print("########### Train Test Val Script started ###########")
    # data_csv = pd.read_csv("DataSet_Final.csv") ##Use if you have classes saved in any .csv file

    # for name in data_csv['names'].unique()[:10]:
    #    classes_dir.append(name)

    for folder_no, class_name in enumerate(class_names):
        # Creating partitions of the data after shuffeling
        print("$$$$$$$ Class Name: " + class_name + " $$$$$$$")
        src = source_dir_path + "/" + class_name  # Folder to copy images from

        names_of_all_files = os.listdir(src)

        if same_file_number:
            # regulates the number of files to be selected for training, testing and validating the model.
            selected_file_names = np.array(names_of_all_files)[:no_files_in_folder]
        else:
            selected_file_names = np.array(names_of_all_files)

        np.random.shuffle(selected_file_names)

        train_file_names, val_file_names, test_file_names = np.split(selected_file_names,
                                                                     [int(selected_file_names.size * (
                                                                             1 - (val_ratio + test_ratio))),
                                                                      int(selected_file_names.size * (
                                                                              1 - test_ratio)),
                                                                      ])

        train_file_names = [src + '//' + name for name in train_file_names.tolist()]
        val_file_names = [src + '//' + name for name in val_file_names.tolist()]
        test_file_names = [src + '//' + name for name in test_file_names.tolist()]

        print(f"Total {class_name} images: {len(names_of_all_files)}")
        print(f"The No of selected {class_name} images is: {selected_file_names.size}")
        print('Training: ' + str(len(train_file_names)))
        print('Validation: ' + str(len(val_file_names)))
        print('Testing: ' + str(len(test_file_names)))

        # check if the folder exists. if it exists delete the folder before recreating it
        # if os.path.exists(destination_dir_path):
        #     shutil.rmtree(destination_dir_path)
        # # Creating Train / Val / Test folders (One time use)
        os.makedirs(destination_dir_path + '/train//' + str(folder_no) + "_" + class_name)
        os.makedirs(destination_dir_path + '/val//' + "" + str(folder_no) + "_" + class_name)
        os.makedirs(destination_dir_path + '/test//' + "" + str(folder_no) + "_" + class_name)

        # Copy-pasting images
        for name in train_file_names:
            shutil.copy(name, destination_dir_path + '/train//' + str(folder_no) + "_" + class_name)

        for name in val_file_names:
            shutil.copy(name, destination_dir_path + '/val//' + str(folder_no) + "_" + class_name)

        for name in test_file_names:
            shutil.copy(name, destination_dir_path + '/test//' + str(folder_no) + "_" + class_name)

    print("########### Train Test Val Script Ended ###########")
    return destination_dir_path


def preprocess_images_from_directory(train_dir_path, test_dir_path
                                     , label_mode, img_size, seed=45,
                                     batch_size=32, val_split=0.2, ):
    """
    preprocess images using the training and testing directories only, a separate validiation dataset is not required.
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


# recompile the model with lower learning rate when fine tuning the best practise
# is to reduce by x10.
# method for fine-tuning a transfer learning feature extracted model.
def fine_tune_created_vanilla_model(model,
                                    train_data,
                                    val_data,
                                    callbacks,
                                    base_model,
                                    training_history,
                                    train_entire_model=False,
                                    no_layer_to_show=3,
                                    layers_to_freeze=-5,
                                    fine_tune_epochs=20,
                                    optimizer=tf.keras.optimizers.Adam(0.0001),
                                    loss="binary_crossentropy",
                                    metrics=["accuracy"],
                                    ):
    """
    :param no_layer_to_show: which layers will be printed as modified.
    :param training_history:
    :param base_model: This is the pre-trained tensorflow model
    :param layers_to_freeze:
    :param train_entire_model:
    :param model: pass the model to be fine-tuned
    :param train_data: tensorflow preprocessed training dataset
    :param val_data: tensorflow preprocessed validation dataset
    :param callbacks: tensorflow callback methods
    :param fine_tune_epochs: the number of epochs the model will be fine-tuned
    :param optimizer: the optimization algorithm that will be used
    :param loss: the loss function that will be used
    :param metrics: metrics for evaluating the model performance
    :return:
     return {
        "model": model,
        "new_history": new_history
    }
    """
    # unfreeze all layers in the base model
    base_model.trainable = True

    if not train_entire_model:
        for layer in base_model.layers[:layers_to_freeze]:
            layer.trainable = False

    # check the trainable layers in our efficientnet base model.
    for num, layer in enumerate(model.layers[no_layer_to_show].layers):
        print(num, layer.name, layer.trainable)
    # Compile and fit the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Fine-tune for 30 more epochs
    # fine_tune_epochs = 20 #because we have added to the intial 5 epochs during featue extraction TL.

    new_history = model.fit(
        train_data,
        epochs=fine_tune_epochs,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        initial_epoch=training_history.epoch[-1],
        callbacks=callbacks
    )

    return {
        "model": model,
        "new_history": new_history
    }
