import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit


# Created this process as a function, just in case we want to do this for multiple songs
def notes_to_octaves(voices: list) -> list:
    """ Transform every note to its corresponding octave """
    octaves = {(i-1): [j+i*12 for j in range(12)] for i in range(11)}  # keys of dict = [-1,0,...,9] as in the table
    del octaves[-1][0]  # remove 0 from the "C" note's values, as it is used for break in our case
    octaves.setdefault(-2, [0])

    notes_transformed = [[], [], [], []]
    for i in range(len(voices)):
        for j in range(len(voices[i])):
            octave_of_note = [key for key, values in octaves.items() if voices[i][j] in values]
            notes_transformed[i].append(octave_of_note[0])

    return notes_transformed


def one_hot_encode_input(raw_input: list, set_raw_input: list) -> list:
    """ One-hot encoding for notes and octaves """
    X_vec_all = []
    for idx, voice in enumerate(raw_input):
        X_vec = []
        for note in voice:
            n = set_raw_input[idx].index(note)
            vec = [1 if i == n else 0 for i in range(len(set_raw_input[idx]))]
            X_vec.append(vec)
        X_vec_all.append(X_vec)

    return X_vec_all


# TODO: This needs to be rewritten, was used only for previous testing with the counting loss
def check_scores(regressor, X_train, X_test, y_train, y_test, set_X, Y) -> None:
    """ Testing score on train and test set with hand-crafted loss function """
    print(f"Score on train set: {regressor.score(X_train, y_train)}")
    print(f"Score on test set: {regressor.score(X_test, y_test)}")

    predictions = []
    correct_predictions = []
    for i in range(len(X_train)):
        yhat = regressor.predict([X_train[i]])
        pitch = set_X[yhat.tolist()[0].index(max(yhat.tolist()[0]))]
        predictions.append(pitch)
        if pitch == Y[i]:  # Y[i] = X_train[i+1], but without the one-hot encoding form
            correct_predictions.append(pitch)
    print(f"Train set size: {len(X_train)}, Correct predictions: {len(correct_predictions)}")
    print(f"Total predictions: {len(predictions)}\n")


data = pd.read_csv("F.txt", sep="\t", header=None)
voices_all = [list(data[i]) for i in range(data.shape[1])]

# 1) Encoding notes
# Transform the notes into an 'octaveless' representation to reduce range of input
# -> e.g. C5, C#3, D2...,B3 becomes 0,1,2,...11
notes = {i: [j*12+i for j in range(11) if j or i] for i in range(12)}  # 0 value is reserved for break, hence the "j or i"
notes.setdefault(-1, [0])  # break

transformed_voices = [[], [], [], []]
for idx_voice, voice in enumerate(voices_all):
    for note in voice:
        for key, values in notes.items():
            if note in values:
                transformed_voices[idx_voice].append(key)
set_transformed_voices = [list(set(voice)) for voice in transformed_voices]
notes_encoded_all_voices = one_hot_encode_input(transformed_voices, set_transformed_voices)

# 2) Encoding octaves
# the encoding of octaves is different for each voice in terms of the real value represented by each binary digit,
# because after plotting the occurrence and histogram of octaves per voice it is clear that they follow a certain range
notes_transformed_to_octaves = notes_to_octaves(voices_all)
set_notes_transformed_to_octaves = [list(set(voice)) for voice in notes_transformed_to_octaves]
octaves_encoded_all_voices = one_hot_encode_input(notes_transformed_to_octaves, set_notes_transformed_to_octaves)

# TODO: Join encodings


# 4) Exploring non-one-hot encoded octaves for recognizing possible patterns
titles = ["Soprano", "Alto", "Tenor", "Bass"]

figure = plt.figure(figsize=(16, 12))
x_axis = [x for x in range(len(notes_transformed_to_octaves[0]))]
for idx, voice in enumerate(notes_transformed_to_octaves):
    plt.subplot(2, 2, idx + 1)
    plt.plot(x_axis, voice)
    plt.xlabel("Time")
    plt.ylabel("Octave")
    plt.ylim(-2, max(max(set_notes_transformed_to_octaves)))
    plt.title(f"Frequency of octaves in voice: {idx} ({titles[idx]})")
plt.close()
figure.savefig(f"Frequency of octaves, with breaks denoted as '-2'.pdf")

figure = plt.figure(figsize=(16, 12))
x_min, x_max = [min(min(set_notes_transformed_to_octaves)), max(max(set_notes_transformed_to_octaves))]
bins_num = len(range(x_min, x_max+1))*10
for idx, voice in enumerate(notes_transformed_to_octaves):
    plt.subplot(2, 2, idx + 1)
    plt.hist(voice, bins=bins_num, range=(x_min, x_max))
    plt.xlabel("Octave")
    plt.title(f"Frequency of octaves in voice: {idx} ({titles[idx]})")
plt.close()
figure.savefig(f"Histograms of octaves, with breaks denoted as '-2'.pdf")


# # Assigning input and output with a window size of 1
# X_t0 = np.array(X_vec[:-1])
# y = np.array(X_vec[1:])
#
# # Split train and test dataa with default 80-20 ratio
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, test_index in tscv.split(X_t0):
#     X_train, X_test = X_t0[train_index], X_t0[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# lin_reg = LinearRegression().fit(X_train, y_train)
# ridge_reg = Ridge(alpha=15).fit(X_train, y_train)
#
#
# check_scores(lin_reg, X_train, X_test, y_train, y_test, set_X, Y)
# check_scores(ridge_reg,  X_train, X_test, y_train, y_test, set_X, Y)
