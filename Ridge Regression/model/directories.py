import os
import shutil


def gendir(voice_num):
    path_start = os.path.abspath('.')
    dirtypes = ["note", "octave"]
    for dir in dirtypes:
        parentdir = f"voice{voice_num}_{dir}"
        os.makedirs(f"{parentdir}\\voice{voice_num}_{dir}_weight1")
        os.makedirs(f"{parentdir}\\voice{voice_num}_{dir}_weight2")
        os.makedirs(f"{parentdir}\\voice{voice_num}_{dir}_weight3")


def organize_pickles(parent_folder):
    start_path = os.path.abspath('.')
    os.chdir(f"{parent_folder}")
    print(os.getcwd())

    target_folders = [f"{parent_folder}_weight1", f"{parent_folder}_weight2", f"{parent_folder}_weight3"]
    print(target_folders[0], target_folders[1], target_folders[2])
    for idx, file in enumerate(os.listdir('.')):
        if file.endswith("weight1.pickle"):
            shutil.copy(file, f"{target_folders[0]}")
        elif file.endswith("weight2.pickle"):
            shutil.copy(file, f"{target_folders[1]}")
        elif file.endswith("weight3.pickle"):
            shutil.copy(file, f"{target_folders[2]}")

    os.chdir(start_path)
    print(os.getcwd())


# organize_pickles("voice1_note", "voice_1_note_weight123")
organize_pickles("voice3_octave")

# gendir(1)
# gendir(2)
# gendir(3)
# gendir(4)
