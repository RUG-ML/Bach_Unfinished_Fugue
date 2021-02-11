import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pprint
from model import Voice
from model import notes_transformed_to_octaves, set_notes_transformed_to_octaves, data, reduced_notes


def plt_notes(reduced=False, predictions=None, voice=None) -> None:
    """ Plot original and reduced notes - voice1 """
    def adjust_yticks():
        if reduced:
            return plt.yticks(np.arange(-1, 12), fontsize=fontsize)
        return plt.yticks(np.arange(0, 81, 10), fontsize=fontsize)

    if reduced:
        notes = reduced_notes[0]
        plt_name = "reduced"
        ylabel = "MIDI encoded and reduced pitch value"
    elif not reduced and not predictions:
        notes = data[0]
        plt_name = "original"
        ylabel = "MIDI encoded pitch value"

    if predictions and voice:
        notes = predictions
        plt_name = "prediction"
        ylabel = "MIDI encoded pitch value"

    pd_notes = pd.DataFrame([notes]).transpose()
    figure = plt.figure(figsize=(16, 12))
    fontsize = 30
    plt.plot(pd_notes)
    adjust_yticks()
    plt.xticks(fontsize=fontsize)
    plt.xlabel("n", fontsize=fontsize + 5)
    plt.ylabel(ylabel, fontsize=fontsize)
    # plt.vlines(3824, ymin=min(pd_notes), ymax=max(pd_notes), color="r", linestyles="solid")
    plt.axvline(3824, color="r")
    if predictions and voice:
        plt.title(f"Voice {voice}", fontsize=fontsize)
        figure.savefig(f"Voice{voice}_{plt_name}.pdf")
    else:
        plt.title(f"Voice 1", fontsize=fontsize)
        figure.savefig(f"Voice1_{plt_name}_notes.pdf")
    plt.close()

    return None


def plt_octave_frequency(voice_num: int, plt_multiple: bool) -> None:
    """ Exploring non-one-hot encoded octaves for recognizing possible patterns """
    fontsize = 30
    figure = plt.figure(figsize=(16, 12))
    x_axis = [x for x in range(len(notes_transformed_to_octaves[0]))]

    # plot all voice_num together
    if plt_multiple:
        for idx, voice in enumerate(notes_transformed_to_octaves[:voice_num]):
            plt.subplot(2, 2, idx + 1)
            plt.plot(x_axis, voice)
            plt.xlabel("n", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.ylabel("Octave", fontsize=fontsize)
            plt.ylim(-2, max(max(set_notes_transformed_to_octaves)))
            plt.title(f"Voice {idx+1}", fontsize=fontsize)
        plt.close()
        figure.savefig(f"Voice {'-'.join([str(x) for x in list(range(1,voice_num+1))])} octaves.pdf")
    else:
        # plot the single, chosen voice
        plt.plot(x_axis, notes_transformed_to_octaves[voice_num-1])
        plt.xlabel("n", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("Octave", fontsize=fontsize)
        plt.ylim(-2, max(max(set_notes_transformed_to_octaves)))
        plt.title(f"Voice {voice_num}", fontsize=fontsize)
        plt.close()
        figure.savefig(f"Voice{voice_num}_octaves.pdf")

    return None


def plt_octave_histogram(voices=4) -> None:
    figure = plt.figure(figsize=(16, 12))
    x_min, x_max = [min(min(set_notes_transformed_to_octaves)), max(max(set_notes_transformed_to_octaves))]
    bins_num = len(range(x_min, x_max+1))*10
    for idx, voice in enumerate(notes_transformed_to_octaves[:voices]):
        plt.subplot(2, 2, idx + 1)
        plt.hist(voice, bins=bins_num, range=(x_min, x_max))
        plt.xlabel("Octave")
        plt.title(f"Voice {idx}")
    plt.close()
    figure.savefig(f"Histograms of octaves, with breaks denoted as '-2'.pdf")

    return None


def plt_flexibility_curve(parent_folder, x_axis_name: str) -> None:
    path = os.path.abspath('.')
    os.chdir(f"..\\{parent_folder}")
    child_folders = [f"{parent_folder}_weight1", f"{parent_folder}_weight2", f"{parent_folder}_weight3"]
    fontsize = 20
    for folder in child_folders:
        os.chdir(folder)
        print(f"Creating pdf in folder {os.getcwd()}...")
        files = os.listdir('.')
        losses_train_normal = []
        losses_train_feedback = []
        losses_test_normal = []
        losses_test_feedback = []
        for i in range(150):
            print(i)
            curr_file = f"{parent_folder}_w{i+1}_{folder.split('_')[-1]}.pickle"
            if curr_file not in files:
                continue
            with open(curr_file, "rb") as file_obj:
                voice_obj = pickle.load(file_obj)
            losses_train_normal.append(voice_obj.cv_losses['train']['normal_avg'])
            # losses_train_feedback.append(voice_obj.cv_losses['train']['feedback_avg'])
            losses_test_normal.append(voice_obj.cv_losses['test']['normal_avg'])
            # losses_test_feedback.append(voice_obj.cv_losses['test']['feedback_avg'])
        # losses = [losses_train_normal, losses_train_feedback, losses_test_normal, losses_test_feedback]
        losses = [losses_train_normal, losses_test_normal]
        print("All files have been read.")
        titles = ["Risks without feedback", "Risks with feedback"]
        risks = ["Empirical Risk", "Expected Risk"]
        figure = plt.figure(figsize=(16, 12))
        # for i in range(2):
        #     plt.subplot(2, 1, i+1)
        plt.plot(losses[0])
        plt.plot(losses[1])
        plt.ylim(ymin=min(min(losses[:])) * 0.9, ymax=max(max(losses[:])) * 1.1)
        plt.xlabel(x_axis_name, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel(f"Average Loss", fontsize=fontsize)
        plt.legend(risks, fontsize=fontsize)
        plt.title(f"{parent_folder.split('_')[0].capitalize()} {parent_folder.split('_')[1]} model performance", fontsize=fontsize)
        plt.close()
        pdf_name = f"Losses - {folder}.pdf"
        figure.savefig(pdf_name)
        print(f"{pdf_name} has been created")
        os.chdir('..')
    os.chdir(path)

    return None


# plt_octave_frequency(2, False)
# plt_octave_frequency(3, False)
# plt_octave_frequency(4, False)
predictions = pd.read_csv("Bach_Ridge.txt", sep=" ", header=None)
plt_notes(predictions=list(predictions[0]), voice=1)
plt_notes(predictions=list(predictions[1]), voice=2)
plt_notes(predictions=list(predictions[2]), voice=3)
plt_notes(predictions=list(predictions[3]), voice=4)
# # plt_notes(reduced=True)

# plt_flexibility_curve('voice1_note', "window_size")
# plt_flexibility_curve('voice1_octave', "window_size")
# plt_flexibility_curve('voice2_note', "window_size")
# plt_flexibility_curve('voice2_octave', "window_size")
# plt_flexibility_curve('voice4_octave', "window_size")
# plt_flexibility_curve('voice3_note', "window_size")
# plt_flexibility_curve('voice3_octave', "window_size")
# plt_flexibility_curve('voice4_note', "window_size")
# plt_flexibility_curve('voice4_octave', "window_size")



