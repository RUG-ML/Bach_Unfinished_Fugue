import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pprint
from typing import Union

from run import Voice
from run import notes_transformed_to_octaves, set_notes_transformed_to_octaves, data, reduced_notes


def plt_notes(reduced=False) -> None:
    """ Plot original and reduced notes - voice1 """
    def adjust_yticks():
        if reduced:
            return plt.yticks(np.arange(-1, 12), fontsize=fontsize)
        return plt.yticks(np.arange(0, 81, 10), fontsize=fontsize)

    if reduced:
        notes = reduced_notes[0]
        plt_name = "reduced"
        ylabel = "MIDI encoded and reduced pitch value"
    else:
        notes = data[0]
        plt_name = "original"
        ylabel = "MIDI encoded pitch value"

    pd_notes = pd.DataFrame([notes]).transpose()
    figure = plt.figure(figsize=(16, 12))
    fontsize = 30
    plt.plot(pd_notes)
    adjust_yticks()
    plt.xticks(fontsize=fontsize)
    plt.xlabel("n", fontsize=fontsize + 5)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title("Voice 1", fontsize=fontsize)
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


def plot_flexibility(losses, fontsize, plot_title, x_axis_name, fig_name) -> None:
    """ Plot the flexibility curves """

    error_types = ["Train Error", "Test error"]
    figure = plt.figure(figsize=(16, 12))
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.ylim(ymin=min(min(losses[:])) * 0.9, ymax=max(max(losses[:])) * 1.1)

    plt.xlabel(x_axis_name, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(f"Averaged Cross-entropy loss", fontsize=fontsize)
    plt.title(plot_title, fontsize=fontsize)

    plt.legend(error_types, fontsize=fontsize)
    plt.close()
    pdf_name = f"Losses - {fig_name}.pdf"
    figure.savefig(pdf_name)
    print(f"{pdf_name} has been created")

    return None


def flexibility_curve(parent_folder, x_axis_name: str, plot_title: str, plotting=True) -> Union[list, None]:
    """ Get flexibility curves for each model type, two lines per plot """

    path = os.path.abspath('.')
    os.chdir(f"..\\{parent_folder}")
    # child_folders = [f"{parent_folder}_weight1", f"{parent_folder}_weight2", f"{parent_folder}_weight3"]
    child_folders = [f"{parent_folder}_weight2"]
    fontsize = 30
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

        if not plotting:
            os.chdir(path)
            return losses
        else:
            plot_flexibility(losses, fontsize, plot_title, x_axis_name, folder)

        os.chdir('..')
    os.chdir(path)

    return None


def joint_flexibility_curves(losses: list) -> None:
    fontsize = 30
    figure = plt.figure(figsize=(16, 12))
    error_types = ["Train error", "Test error"]
    colors = ["blue", "orange"]
    for idx_l, loss_type in enumerate(losses):
        for idx_v, voice in enumerate(loss_type):
            plt.plot(voice, color=colors[idx_l])
        
    plt.xlabel("window size", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(f"Averaged Cross-entropy loss", fontsize=fontsize)
    plt.title("Singers 1-4: Note model performances", fontsize=fontsize)
    plt.ylim(ymin=0, ymax=0.4)

    plt.legend(error_types, fontsize=30, loc="upper right")
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(colors[0])
    leg.legendHandles[1].set_color(colors[1])

    plt.close()
    pdf_name = f"Losses - notes, all voices.pdf"
    figure.savefig(pdf_name)

    return None


# plt_octave_frequency(2, False)
# plt_octave_frequency(3, False)
# plt_octave_frequency(4, False)
# # plt_notes()
# # plt_notes(reduced=True)

[v1_train_loss, v1_test_loss] = flexibility_curve('voice1_note', "window size",
                                                  "First voice note model performance", plotting=False)
# flexibility_curve('voice1_octave', "window size", "First voice octave note model performance")
[v2_train_loss, v2_test_loss] = flexibility_curve('voice2_note', "window size",
                                                  "Second voice note model performance", plotting=False)
# flexibility_curve('voice2_octave', "window size", "Second voice octave note model performance")
[v3_train_loss, v3_test_loss] = flexibility_curve('voice3_note', "window size",
                                                  "Third voice note model performance", plotting=False)
# flexibility_curve('voice3_octave', "window size", "Third voice octave note model performance")
[v4_train_loss, v4_test_loss] = flexibility_curve('voice4_note', "window size",
                                                  "Fourth voice note model performance", plotting=False)
# plt_flexibility_curve('voice4_octave', "window size", "Fourth voice octave model performance")


joint_flexibility_curves([[v1_train_loss, v2_train_loss, v3_train_loss, v4_train_loss],
                         [v1_test_loss, v2_test_loss, v3_test_loss, v4_test_loss]])
