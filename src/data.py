# -*- coding: utf-8 -*-\
from zipfile import ZipFile
from pathlib import Path
import shutil
import splitfolders
import os


def unzip():
    """Extrai o data.zip no raiz do projeto"""
    data_zip_path = os.path.join(str(Path().absolute()), "data.zip")
    with ZipFile(data_zip_path, "r") as zipObj:
        zipObj.extractall("data")
        print("File is unzipped in data folder")


# CREATE TRAIN, VAL, TEST DATA FOLDER
def split_data(
    train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
):
    """split data/* into data/train/data, data/val/data, data/test/data

    Args:
        train_ratio (float, optional): Defaults to 0.7.
        val_ratio (float, optional): Defaults to 0.15.
        test_ratio (float, optional): Defaults to 0.15.
    """
    print(
        f'Spliting data into: \
        train_ratio"{train_ratio} \
        val_ratio"{val_ratio} \
        test_ratio"{test_ratio}'
    )
    splitfolders.ratio(
        "data",
        output="output",
        seed=42,
        ratio=(train_ratio, val_ratio, test_ratio),
        group_prefix=None,
    )


def move_data_files():
    """ Split folder creates a subfolder 'data' inside every folder
        this moves every image to data parent /train, /val, /test folder
    """
    print("Moving images to train, val and test folder")
    for folder in os.listdir(os.path.join(str(Path().absolute()), "output")):
        folder_path = os.path.join(str(Path().absolute()), "output", folder)
        print("------------------")
        print(folder_path)
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            print(sub_folder_path)
            for file in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, file)
                new_file_path = os.path.join(folder_path, file)
                shutil.move(str(file_path), str(folder_path))


def remove_subfolder_data():
    """Remove data/../data created on splitfolder"""
    print("Removing data subfolder")
    for folder in os.listdir(os.path.join(str(Path().absolute()), "output")):
        folder_path = os.path.join(str(Path().absolute()), "output", folder, "data")
        print(f"Removing {folder_path} folder")
        shutil.rmtree(str(folder_path))


def create_class_folder():
    """Create class_a and class_b folder based on filename
        {...}0.jpeg = class_a
        {...}1.jpeg = class_b
    """
    print("Create class folder")
    for folder in os.listdir(os.path.join(str(Path().absolute()), "output")):
        folder_path = os.path.join(str(Path().absolute()), "output", folder)
        print("------------------")
        print(folder_path)
        class_a_path = os.path.join(folder_path, "class_a")
        class_b_path = os.path.join(folder_path, "class_b")
        try:
            os.mkdir(class_a_path)
        except OSError:
            print("Creation of the directory %s failed" % class_a_path)
        else:
            print("Successfully created the directory %s " % class_a_path)

        try:
            os.mkdir(class_b_path)
        except OSError:
            print("Creation of the directory %s failed" % class_b_path)
        else:
            print("Successfully created the directory %s " % class_b_path)

        for file in os.listdir(folder_path):
            if file not in ["class_a", "class_b"]:
                file_path = os.path.join(folder_path, file)
                if "0.jpg" in file_path:
                    new_file_path = os.path.join(folder_path, "class_a", file)
                    shutil.move(str(file_path), str(new_file_path))
                else:
                    new_file_path = os.path.join(folder_path, "class_b", file)
                    shutil.move(str(file_path), str(new_file_path))


def rename_data_folder():
    """Delete data folder and rename output to data
    """
    print("Renaming data folder")
    shutil.rmtree(os.path.join(str(Path().absolute()), "data"))
    os.rename(
        os.path.join(str(Path().absolute()), "output"),
        os.path.join(str(Path().absolute()), "data"),
    )


if __name__ == "__main__":
    unzip()
    split_data()
    move_data_files()
    remove_subfolder_data()
    create_class_folder()
    rename_data_folder()

