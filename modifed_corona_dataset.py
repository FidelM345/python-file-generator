# import required module
import os
import shutil


def modify_corona_dataset(directory_path):
    """
    :param (String) directory_path: provides the path of the directory to be modified
    :return: null (does not return anything)
    :info -> used for converting this kaggle dataset into a format that I can easily use im y notebook
    :kaggle link -> https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
    """
    # iterate over files in
    # that directory
    values = []
    # Iterates through the provided folder path and adds all the existing directory paths into a list of values.
    for root, dirs, files in os.walk(directory_path):

        for directory_name in dirs:
            values.append(f"{root}/{directory_name}")
            print(f"{root}/{directory_name}")
            print("\n")
            # for filename in files:
            #     print(os.path.join(root, filename))

    # sorts out and adds the masks' subdirectory into the list
    folder_masks = [item for item in values if 'masks' in item]
    print(f"mask folder: {folder_masks}")
    # Deletes the  masks' subdirectory
    for folder_path in folder_masks:
        try:
            shutil.rmtree(folder_path)  # used for deleting non-empty folders
        except OSError as e:
            print("Error: %s : %s" % (directory_path, e.strerror))

    folder_images = [item for item in values if 'images' in item]
    print(f"images folder: {folder_images}")

    folder_class = [item for item in values if not 'images' in item]
    print(f"class folder: {folder_class}")

    # moving images from the image folder to the parent folder of the respective class.
    for count, source in enumerate(folder_images):
        # fetch all files
        for file_name in os.listdir(source):
            # construct full file path
            source_path = source + "/" + file_name
            destination_path = folder_class[count] + "/" + file_name
            # move only files
            if os.path.isfile(source_path):
                shutil.move(source_path, destination_path)
                # print('Moved:', file_name)

    for folder_path in folder_images:
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print("Error: %s : %s" % (directory_path, e.strerror))
