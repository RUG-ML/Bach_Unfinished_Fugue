import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, copy
import pprint

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit


class Voice:
    """ Class for each voice to make all model-related data corresponding to each voice easily accessible """
    def __init__(self, name):
        self.name = name  # dummy property to make the objects explicitly distinguishable

    # TODO
    def check_scores(self):
        return


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


def fit_model_to_voice(voice: object, voice_encoded: list, transformed_voice: list) -> object:
    # Assigning input and output with a window size of 1
    voice.X = np.array(voice_encoded[:-1])
    voice.y_encoded = np.array(voice_encoded[1:])
    voice.y_raw = np.array(transformed_voice[1:])

    # TODO: Implement CV properly, maybe even LOOCV if possible
    # Split train and test data with default 80-20 ratio
    voice.tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in voice.tscv.split(voice.X):
        voice.X_train, voice.X_test = voice.X[train_index], voice.X[test_index]
        voice.y_train, voice.y_test = voice.y_encoded[train_index], voice.y_encoded[test_index]

    # TODO: Implement sklearn's grid-search for finding out window size and alpha for Ridge
    voice.lin_reg = LinearRegression()
    voice.lin_reg.fit(voice.X_train, voice.y_train)

    voice.ridge_reg = Ridge(alpha=1)
    voice.ridge_reg.fit(voice.X_train, voice.y_train)

    return voice


def get_n_likely_indices(predictions_vec: list, n: int) -> list:
    """ Retrieves the 'n' most probable indices of the model's output """
    indices = []
    # creating copy to assure the "get_n_likely" indices doesn't mess things up
    vec_copy = copy.deepcopy(list(predictions_vec))
    for i in range(n):
        max_idx = vec_copy.index(max(vec_copy))
        indices.append(max_idx)
        vec_copy[max_idx] = min(vec_copy)
    return indices


# 0) Read data
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
# 3)

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


voice_1_note = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], notes_transformed_to_octaves[0])
voice_1_octave = fit_model_to_voice(Voice("Soprano octaves"), octaves_encoded_all_voices[0], notes_transformed_to_octaves[0])

voice_2_note = fit_model_to_voice(Voice("Alto notes"), notes_encoded_all_voices[1], notes_transformed_to_octaves[1])
voice_2_octave = fit_model_to_voice(Voice("Alto octaves"), octaves_encoded_all_voices[1], notes_transformed_to_octaves[1])

voice_3_note = fit_model_to_voice(Voice("Tenor notes"), notes_encoded_all_voices[2], notes_transformed_to_octaves[2])
voice_3_octave = fit_model_to_voice(Voice("Tenor octaves"), octaves_encoded_all_voices[2], notes_transformed_to_octaves[2])

voice_4_note = fit_model_to_voice(Voice("Bass notes"), notes_encoded_all_voices[3], notes_transformed_to_octaves[3])
voice_4_octave = fit_model_to_voice(Voice("Bass octaves"), octaves_encoded_all_voices[3], notes_transformed_to_octaves[3])


# This can be be deleted later on,
# just a sanity check whether the object oriented approach produces proper values or not
# ---------------------------------------------------------------------------------------------------------------------
voices_note = [voice_1_note, voice_2_note, voice_3_note, voice_4_note]
voices_octave = [voice_1_octave, voice_2_octave, voice_3_octave, voice_4_octave]
for i in range(len(voices_note)):
    print(f"Checking prediction values for voice {i+1}...")
    print(f"Train score on notes: {voices_note[i].lin_reg.score(voices_note[i].X_train, voices_note[i].y_train)}")
    print(f"Test score on notes: {voices_note[i].lin_reg.score(voices_note[i].X_test, voices_note[i].y_test)}")
    print(f"Train score on octaves: {voices_octave[i].lin_reg.score(voices_octave[i].X_train, voices_octave[i].y_train)}")
    print(f"Test score on octaves: {voices_octave[i].lin_reg.score(voices_octave[i].X_test, voices_octave[i].y_test)}")
    print("-------------------------------------------------------------------------\n")
# ---------------------------------------------------------------------------------------------------------------------

# TODO: Finish this
# Testing suggested loss function for one prediction

# a) emphasize and deemphasize large and small values
test_pred = np.power(voice_2_note.lin_reg.predict([voice_2_note.X_train[578]]), 3)

# b) normalize values so they sum to 1
normalization_denominator = sum(test_pred[0])
normalized_preds = [pred/normalization_denominator for pred in test_pred[0]]

# c) get 'n' number of indices that have a high probability
# likely_indices = get_n_likely_indices(normalized_preds, 3)
# normalized_preds.sort(reverse=True)
# chosen_preds = normalized_preds[:len(likely_indices)]
#
# # d) randomization of final choice
# probability_idx_dict = {idx: prob for idx, prob in zip(likely_indices, chosen_preds)}
# probability_idx_keys = list(probability_idx_dict.keys())
# random.shuffle(probability_idx_keys)
# final_prediction = probability_idx_dict[probability_idx_keys[0]]
# print(final_prediction)
