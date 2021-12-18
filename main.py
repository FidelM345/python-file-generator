import os
import numpy as np
import shutil

# import pandas as pd
import os
import numpy as np
import shutil


# import pandas as pd
def train_test_split(source_dir_path, destination_dir_path, class_names, val_ratio, test_ratio,
                     same_file_number=False, no_files_in_folder=1345):
    """
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

        # if classes_data_should_be_same:
        #     train_file_names, val_file_names, test_file_names = np.split(selected_file_names,
        #                                                                  [int(selected_file_names.size * (
        #                                                                          1 - (val_ratio + test_ratio))),
        #                                                                   int(selected_file_names.size * (
        #                                                                           1 - test_ratio)),
        #                                                                   ])

        # else:
        #     train_file_names, val_file_names, test_file_names = np.split(np.array(names_of_all_files),
        #                                                                  [int(len(names_of_all_files) * (
        #                                                                          1 - (val_ratio + test_ratio))),
        #                                                                   int(len(names_of_all_files) * (
        #                                                                           1 - test_ratio)),
        #                                                                   ])

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


class_names_list = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
destination_folder = train_test_split("covid19-radiography-database", "categorized_corona_data", class_names_list, 0.1,
                                      0.2, True)
print(f"{destination_folder}")
