import pandas as pd
import numpy as np
import random
import copy
import math
import pprint
from typing import Union

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit


class Voice:
    """ Class for each voice to make all model-related data corresponding to each voice easily accessible """
    def __init__(self, name):
        self.name = name  # dummy property to make the objects explicitly distinguishable


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


def one_hot_encode(raw_input: Union[list, int], set_raw_input: list) -> list:
    """ One-hot encoding for notes and octaves """
    X_vec = []

    if type(raw_input) == list:
        for note in raw_input:
            n = set_raw_input.index(note)
            vec = [1 if i == n else 0 for i in range(len(set_raw_input))]
            X_vec.append(vec)
    else:
        n = set_raw_input.index(raw_input)
        vec = [1 if i == n else 0 for i in range(len(set_raw_input))]
        return vec

    return X_vec


def window_size(voice_encoded: list, size: int):
    windowed_data = [list(np.array(voice_encoded[i:(i+size)]).flatten()) for i in range(len(voice_encoded)-size)]

    return windowed_data


def cv_loss(voice: object, cv_count: int, iter_num: int):

    loss_train_normal = check_loss(voice, "train")
    loss_test_normal = check_loss(voice, "test")
    loss_train_feedback = 0
    loss_test_feedback = 0
    for i in range(iter_num):
        loss_train_feedback += check_loss(voice, "train", True)
        loss_test_feedback += check_loss(voice, "test", True)
    loss_train_feedback /= iter_num
    loss_test_feedback /= iter_num

    voice.cv_losses['train']['normal'].append({f'CV_{cv_count}': loss_train_normal})
    voice.cv_losses['train']['feedback'].append({f'CV_{cv_count}': loss_train_feedback})
    voice.cv_losses['test']['normal'].append({f'CV_{cv_count}': loss_test_normal})
    voice.cv_losses['test']['feedback'].append({f'CV_{cv_count}': loss_test_feedback})

    voice.cv_losses['train']['normal_avg'] = np.mean(np.array([list(cv.values())[0] for cv in voice.cv_losses['train']['normal']]))
    voice.cv_losses['train']['feedback_avg'] = np.mean(np.array([list(cv.values())[0] for cv in voice.cv_losses['train']['feedback']]))

    voice.cv_losses['test']['normal_avg'] = np.mean(np.array([list(cv.values())[0] for cv in voice.cv_losses['test']['normal']]))
    voice.cv_losses['test']['feedback_avg'] = np.mean(np.array([list(cv.values())[0] for cv in voice.cv_losses['test']['feedback']]))


def fit_model_to_voice(voice: object, voice_encoded: list, transformed_voice: list, size: int) -> object:

    voice.X_encoded = voice_encoded
    voice.y_midi = np.array(transformed_voice[size:])  # it's in the format of the keys of the "notes" dict
    voice.X_encoded_windowed = np.array(window_size(voice_encoded, size))
    voice.y_encoded_windowed = np.array(voice_encoded[size:])

    # Split train and test data with default 80-20 ratio
    voice.cv_losses = {'train': {'normal': [], 'feedback': []}, 'test': {'normal': [], 'feedback': []}}
    cv_count = 0
    splits = 5
    voice.tscv = TimeSeriesSplit(n_splits=splits)
    for train_index, test_index in voice.tscv.split(voice.X_encoded_windowed):
        cv_count += 1
        voice.X_train, voice.X_test = voice.X_encoded_windowed[train_index], voice.X_encoded_windowed[test_index]
        voice.y_train, voice.y_test = voice.y_encoded_windowed[train_index], voice.y_encoded_windowed[test_index]

        # TODO: Implement sklearn's grid-search for finding out window size and alpha for Ridge
        voice.lin_reg = LinearRegression()
        voice.lin_reg.fit(voice.X_train, voice.y_train)

        voice.ridge_reg = Ridge(alpha=1)
        voice.ridge_reg.fit(voice.X_train, voice.y_train)

        cv_loss(voice, cv_count, 10)

    return voice


def get_n_likely_indices(predictions_vec: list, n: int) -> list:
    """ Retrieves the 'n' most probable predictions' indices of the model's output """
    indices = []
    # creating copy to assure the "get_n_likely" indices doesn't mess things up
    vec_copy = copy.deepcopy(list(predictions_vec))
    for i in range(n):
        max_idx = vec_copy.index(max(vec_copy))
        indices.append(max_idx)
        vec_copy[max_idx] = min(vec_copy)
    return indices


def predict_note(voice: object, note: int, test_type: str) -> [int, list]:
    """ Prediction function for both notes and octaves."""
    # a) emphasize and deemphasize large and small values
    if test_type == "train":
        # weighted_preds = np.float_power(voice.lin_reg.predict([voice.X_train[note]]), 1.0)
        weighted_preds = np.power(voice.lin_reg.predict([voice.X_train[note]]), 1)
        # weighted_preds = np.exp(voice.lin_reg.predict([voice.X_train[note]]))
    elif test_type == "test":
        # weighted_preds = np.float_power(voice.lin_reg.predict([voice.X_test[note]]), 1.0)
        weighted_preds = np.power(voice.lin_reg.predict([voice.X_test[note]]), 1)
        # weighted_preds = np.exp(voice.lin_reg.predict([voice.X_test[note]]))
    else:
        raise TypeError("predict_note() - Wrong test type was given. Possible options: ['train', 'test'].")

    # b) normalize values so they sum to 1
    weighted_preds = [pred if pred > 0 else 1e-100 for pred in list(weighted_preds[0])]
    normalized_preds = [pred / sum(weighted_preds) for pred in weighted_preds]

    # c) get 'n' number of indices that have a high probability
    likely_indices = get_n_likely_indices(normalized_preds, 3)
    norm_preds_sorted = sorted(normalized_preds, reverse=True)
    chosen_preds = norm_preds_sorted[:len(likely_indices)]
    remainder = 1 - sum(chosen_preds)

    # d/1) split interval of 0:1 to n sub-intervals according to weights...
    interval_split = [chosen_preds[i] + sum(chosen_preds[:i]) if i > 0 else chosen_preds[i] for i in
                      range(len(chosen_preds))]
    interval_split.append(interval_split[-1] + remainder)

    # d/2) ...make random choice, get the index of the interval which is closest to the guessed value
    rnd_pick = random.random()
    closest_intrv = interval_split.index(min(interval_split, key=lambda x: abs(x - rnd_pick)))
    if interval_split[closest_intrv] < rnd_pick:
        if closest_intrv+1 >= len(likely_indices):  # remainder's probability, a.k.a. most unlucky case
            final_pred = random.choice(list(notes.keys()))
        else:
            final_pred = list(notes.keys())[likely_indices[closest_intrv + 1]]
    else:
        if closest_intrv >= len(likely_indices):
            final_pred = random.choice(list(notes.keys()))
        else:
            final_pred = list(notes.keys())[likely_indices[closest_intrv]]

    return [final_pred, normalized_preds]


def check_loss(voice: object, test_type: str, feedback=False) -> float:

    # make a copy of the object to avoid overwriting
    voice_copy = copy.deepcopy(voice)

    if test_type == "train":
        data_set = voice_copy.X_train
        labels = voice_copy.y_train
    elif test_type == "test":
        data_set = voice_copy.X_test
        labels = voice_copy.y_test
    else:
        raise TypeError("check_score() - Wrong test type was given. Possible options: ['train', 'test'].")

    losses = []
    # check for all notes
    if not feedback:
        for i in range(len(data_set)):
            y_idx = labels[i].argmax()  # index of the single "1" in the binary vector
            [yhat, normalized_preds] = predict_note(voice_copy, i, test_type)
            loss = np.log(normalized_preds[y_idx])  # take log of the winning index
            losses.append(loss)
    else:
        for i in range(len(data_set)):
            y_idx = labels[i].argmax()
            [yhat, normalized_preds] = predict_note(voice_copy, i, test_type)

            # (index of "set_reduced_notes" doesn't matter, all voices include the whole range of notes.keys() anyway)
            yhat_encoded = one_hot_encode(yhat, set_reduced_notes[0])

            # append output back to dataset
            if i < len(data_set)-1:
                # print(data_set[i + 1])
                win_s = len(data_set[i])//13
                data_set[i+1] = np.append(data_set[i][-(win_s-1)*13:], np.array(yhat_encoded))
                # print(yhat_encoded)
                # print(data_set[i + 1], "-------------", sep="\n")

            loss = np.log(normalized_preds[y_idx])  # take log of the winning index
            losses.append(loss)

    return -sum(losses)/len(losses)


# 0) Read data
data = pd.read_csv("model\\F.txt", sep="\t", header=None)
voices_all = [list(data[i]) for i in range(data.shape[1])]

# 1) Encoding notes
# Transform the notes into an 'octaveless' representation to reduce range of input
# -> e.g. C5, C#3, D2...,B3 becomes 0,1,2,...11
notes = {i: [j*12+i for j in range(11) if j or i] for i in range(12)}  # 0 value is reserved for break, hence the "j or i"
notes.setdefault(-1, [0])  # break

reduced_notes = [[], [], [], []]
for idx_voice, voice in enumerate(voices_all):
    for note in voice:
        for key, values in notes.items():
            if note in values:
                reduced_notes[idx_voice].append(key)
set_reduced_notes = [list(set(voice)) for voice in reduced_notes]

notes_encoded_all_voices = []
for i in range(4):
    notes_encoded_all_voices.append(one_hot_encode(reduced_notes[i], set_reduced_notes[i]))

# 2) Encoding octaves
# the encoding of octaves is different for each voice in terms of the real value represented by each binary digit,
# because after plotting the frequency and histogram of octaves per voice it is clear that they follow a certain range
notes_transformed_to_octaves = notes_to_octaves(voices_all)
set_notes_transformed_to_octaves = [list(set(voice)) for voice in notes_transformed_to_octaves]
octaves_encoded_all_voices = []
for i in range(4):
    octaves_encoded_all_voices.append(one_hot_encode(notes_transformed_to_octaves[i], set_notes_transformed_to_octaves[i]))

# 3) Test model with window size "n"
voice1_note_w2 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 2)
voice1_note_w3 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 3)
voice1_note_w5 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 5)
voice1_note_w10 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 10)
voice1_note_w20 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 20)
voice1_note_w50 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 50)
voice1_note_w100 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 100)
voice1_note_w200 = fit_model_to_voice(Voice("Soprano notes"), notes_encoded_all_voices[0], reduced_notes[0], 200)

# print("Calculating empirical risk and risk...", "--------------------------", sep="\n")
# for i in range(5):
#     print(f"Run {i+1}:")
#     print(f"Train set: {check_loss(voice_1_note_10, 'train')}")
#     print(f"Train set(f): {check_loss(voice_1_note_10, 'train', True)}")
#     print(f"Test set: {check_loss(voice_1_note_10, 'test')} |")
#     print(f"Test set(f): {check_loss(voice_1_note_10, 'test', True)} ")
#     print()
# print("--------------------------")

# pprint.pprint(predict_note(voice_1_note_50, 100, "train"))
# pprint.pprint(predict_note(voice_1_note_50, 200, "train"))
# pprint.pprint(predict_note(voice_1_note_50, 100, "test"))
# pprint.pprint(predict_note(voice_1_note_50, 200, "test"))
