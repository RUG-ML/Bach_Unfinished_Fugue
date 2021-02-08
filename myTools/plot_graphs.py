import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pprint
from model.run import Voice
from model.run import notes_transformed_to_octaves, set_notes_transformed_to_octaves, data, reduced_notes


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


def plt_flexibility_curve(folder_name, x_axis_name: str) -> None:
    losses_train_normal = []
    losses_train_feedback = []
    losses_test_normal = []
    losses_test_feedback = []

    interest = []
    path = "C:\\Users\\varga\\Google Drive\\Hollandia\\Groningen\\School\\Courses\\1B\\Machine Learning" \
           "\\PROJECT\\CODE\\GITHUB\\Peter\\Bach_Unfinished_Fugue\\myTools"
    os.chdir(f"pickles\\{folder_name}")
    files = os.listdir('.')
    for i in range(len(files)):
        if not files[i].startswith('voice'):
            continue
        with open(files[i], "rb") as file_obj:
            voice_obj = pickle.load(file_obj)
        losses_train_normal.append(voice_obj.cv_losses['train']['normal_avg'])
        losses_train_feedback.append(voice_obj.cv_losses['train']['feedback_avg'])
        losses_test_normal.append(voice_obj.cv_losses['test']['normal_avg'])
        losses_test_feedback.append(voice_obj.cv_losses['test']['feedback_avg'])
    losses = [losses_train_normal, losses_train_feedback, losses_test_normal, losses_test_feedback]

    titles = ["Empirical Risk", "Empirical Risk with feedback", "Risk", "Risk with feedback"]
    figure = plt.figure(figsize=(16, 12))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(losses[i])
        plt.xlabel(x_axis_name)
        plt.ylabel(f"Loss")
        if i == 0 or i == 2:
            plt.ylim(ymin=0, ymax=100)
        else:
            plt.ylim(ymin=0, ymax=200)
        plt.title(f"{titles[i]}")
    plt.close()
    voice_idx = folder_name.index('voice')
    voice_type = folder_name.split('_')[2]
    model_type = folder_name.split('_')[0].capitalize() + "Reg"
    figure.savefig(f"Losses - {voice_type} - {folder_name[voice_idx:(voice_idx+len('voice')+1)]} - {model_type}.pdf")
    print("Done")
    os.chdir(path)
    print(os.getcwd())

    return None


plt_octave_frequency(2, False)
plt_octave_frequency(3, False)
plt_octave_frequency(4, False)
# plt_notes()
# plt_notes(reduced=True)

# if __name__ == "__main__":
#     # folders = ['lin_reg\\voice1_note_weigth(1)', 'lin_reg\\voice1_octave_weigth(1)',
#     #            'ridge_reg\\voice1_note_weigth(1)', 'ridge_reg\\voice1_octave_weigth(1)',
#     #            'ridge_reg\\voice2_note_weigth(1)', 'ridge_reg\\voice2_octave_weigth(1)',
#     #            'ridge_reg\\voice3_note_weigth(1)', 'ridge_reg\\voice3_octave_weigth(1)',
#     #            'ridge_reg\\voice4_note_weigth(1)', 'ridge_reg\\voice4_octave_weigth(1)',
#     #            'ridge_reg\\voice1_note_weigth(2)', 'ridge_reg\\voice1_octave_weigth(2)']
#     # for folder in folders:
#     #     flexibility_curve('ridge_reg\\voice1_note_weigth(2)', "window_size")
#     flexibility_curve('ridge_reg\\voice1_note_weigth(1)_2', "window_size")

