import pandas as pd
import numpy as np
import random
import copy
import pprint
import pickle
import os
import warnings
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Union, Callable

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")


def create_save_pickle(model_type: str, voice_num: int) -> Callable:
    """ A tool to save and generate models, with varying window-sizes, into the ROM instead of overloading the RAM """
    def wrapper_func():
        for i in range(150):
            voice_obj = fit_model_to_voice(Voice(model_type, voice_num, i+1), cv=True, regularization=True, learning=True)
            if not voice_obj:
                print(f"voice{voice_num+1} {model_type} {i+1} has been skipped due to 'LinAlgError: SVD could not converge.'")
                continue

            with open(os.path.join(os.path.abspath('.'), f"voice{voice_num+1}_{model_type}_w{i+1}_weight1.pickle"), "wb") as file_obj:
                pickle.dump(voice_obj, file_obj)
            print(f"voice{voice_num+1} {model_type} {i+1} has been successfully generated")
    return wrapper_func


class Voice:
    """ Class for each voice to make all model-related data easily accessible and distinguishable """
    def __init__(self, model_type, voice_num, win_size):
        self.model_type = model_type
        self.voice_num = voice_num
        self.win_size = win_size
        if self.model_type == "note":
            self.vec_size = 13
            self.X_encoded = notes_encoded_all_voices[self.voice_num]
            self.reduced_notes = reduced_notes[self.voice_num]
            self.set_reduced_notes = set_reduced_notes[self.voice_num]
        elif self.model_type == "octave":
            self.vec_size = 4
            self.X_encoded = octaves_encoded_all_voices[self.voice_num]
            self.reduced_notes = notes_transformed_to_octaves[self.voice_num]
            self.set_reduced_notes = set_notes_transformed_to_octaves[self.voice_num]


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


def one_hot_encode(reduced_note: Union[list, int], set_reduced_notes: list) -> list:
    """ One-hot encoding for notes and octaves
    @param reduced_note: Note(s)/octave(s) after dimension reduction on the original note(s) of the corresponding voice
    @param set_reduced_notes: The set of notes/octaves that determine the indices for encoding
    @return: A single list or a list of lists of binary encoded note(s)/octave(s)
    """
    X_vec = []
    if type(reduced_note) == list:
        for note in reduced_note:
            n = set_reduced_notes.index(note)
            vec = [1 if i == n else 0 for i in range(len(set_reduced_notes))]
            X_vec.append(vec)
    else:
        n = set_reduced_notes.index(reduced_note)
        vec = [1 if i == n else 0 for i in range(len(set_reduced_notes))]
        return vec
    return X_vec


def window_size(voice_encoded: list, size: int):
    """ Transforms the original, binary encoded n-dimensional voice, regardless of note or octave type, into a new,
    n*size-dimensional voice
    """
    windowed_data = [list(np.array(voice_encoded[i:(i+size)]).flatten()) for i in range(len(voice_encoded)-size)]
    return windowed_data


def cv_loss(voice: object):

    loss_train_normal = check_loss(voice, "train")
    loss_test_normal = check_loss(voice, "test")
    loss_train_feedback = check_loss(voice, "train", True)
    loss_test_feedback = check_loss(voice, "test", True)

    if not any([loss_train_normal, loss_test_normal, loss_train_feedback, loss_test_feedback]):
        voice.cv_losses = None
        return

    voice.cv_losses['train']['normal'].append(loss_train_normal)
    voice.cv_losses['train']['feedback'].append(loss_train_feedback)
    voice.cv_losses['test']['normal'].append(loss_test_normal)
    voice.cv_losses['test']['feedback'].append(loss_test_feedback)
    voice.cv_losses['train']['normal_avg'] = np.mean(voice.cv_losses['train']['normal'])
    voice.cv_losses['train']['feedback_avg'] = np.mean(voice.cv_losses['train']['feedback'])
    voice.cv_losses['test']['normal_avg'] = np.mean(voice.cv_losses['test']['normal'])
    voice.cv_losses['test']['feedback_avg'] = np.mean(voice.cv_losses['test']['feedback'])


def fit_model_to_voice(voice: object, cv=False, regularization=False, learning=False) -> object:

    voice.y_midi = np.array(voice.reduced_notes[voice.win_size:])  # it's in the format of the keys of the "notes" dict
    voice.X_encoded_windowed = np.array(window_size(voice.X_encoded, voice.win_size))
    voice.y_encoded_windowed = np.array(voice.X_encoded[voice.win_size:])

    # always set learning=True if you want to make predictions on the training/test set
    # the learning=False clause is trained on the entire dataset
    if learning:
        voice.cv_losses = {'train': {'normal': [], 'feedback': []}, 'test': {'normal': [], 'feedback': []}}
        voice.tscv = TimeSeriesSplit(n_splits=5)
        voice.cv_alphas = []
        # Split train and test data with default 80-20 ratio
        for train_index, test_index in voice.tscv.split(voice.X_encoded_windowed):
            voice.X_train, voice.X_test = voice.X_encoded_windowed[train_index], voice.X_encoded_windowed[test_index]
            voice.y_train, voice.y_test = voice.y_encoded_windowed[train_index], voice.y_encoded_windowed[test_index]
            if not regularization:
                voice.lin_reg = LinearRegression(fit_intercept=True)
                try:
                    voice.lin_reg.fit(voice.X_train, voice.y_train)
                except np.linalg.LinAlgError:
                    return None
            else:
                try:
                    voice.ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(voice.X_train, voice.y_train)
                except np.linalg.LinAlgError:
                    return None
            if cv:
                cv_loss(voice)
    else:
        voice.ridge_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(voice.X_encoded_windowed, voice.y_encoded_windowed)
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


def predict_note(voice: object, note: int, dataset_type: str, randomization=None, weight=3) -> [int, list]:
    """ Prediction function for both notes and octaves."""
    # a) emphasize and deemphasize large and small values
    if dataset_type == "train":
        weighted_preds = np.float_power(voice.ridge_reg.predict([voice.X_train[note]]), weight)
    elif dataset_type == "test":
        weighted_preds = np.float_power(voice.ridge_reg.predict([voice.X_test[note]]), weight)
    elif dataset_type == "all":
        weighted_preds = np.float_power(voice.ridge_reg.predict([voice.X_encoded_windowed[note]]), weight)
    else:
        raise TypeError("predict_note() - Wrong test type was given. Possible options: ['train', 'test'].")

    # b) normalize values so they sum to 1
    weighted_preds = np.where(weighted_preds < 0, 0, weighted_preds)
    sum_wp = np.sum(weighted_preds)
    normalized_preds = weighted_preds / sum_wp

    # c) get 'n' number of indices that have a high probability
    if not randomization:
        randomization = 3
    assert randomization <= voice.vec_size

    likely_indices = get_n_likely_indices(normalized_preds[0], randomization)
    norm_preds_sorted = sorted(normalized_preds[0], reverse=True)
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
            final_pred = random.choice(voice.set_reduced_notes)
        else:
            final_pred = voice.set_reduced_notes[likely_indices[closest_intrv + 1]]
    else:
        if closest_intrv >= len(likely_indices):
            final_pred = random.choice(voice.set_reduced_notes)
        else:
            final_pred = voice.set_reduced_notes[likely_indices[closest_intrv]]
    return [final_pred, normalized_preds[0]]


def check_loss(voice: object, dataset_type: str, feedback=False) -> Union[float, list]:

    # make a copy of the object to avoid overwriting in case of multiple runs of function with same object
    voice_copy = copy.deepcopy(voice)
    weights = [1]
    losses = []
    losses_all_weights = []
    for weight in weights:
        voice_local = copy.deepcopy(voice_copy)
        if dataset_type == "train":
            data_set_local = voice_local.X_train
            labels_local = voice_local.y_train
        elif dataset_type == "test":
            data_set_local = voice_local.X_test
            labels_local = voice_local.y_test
        else:
            raise TypeError("check_score() - Wrong test type was given. Possible options: ['train', 'test'].")
        for i in range(len(data_set_local)):
            y = labels_local[i]
            yhat = predict_note(voice_local, i, dataset_type, weight=weight)
            if type(yhat) == str:
                return 0
            if feedback:
                yhat_encoded = one_hot_encode(yhat[0], voice.set_reduced_notes)
                if i < len(data_set_local)-1:
                    if voice.win_size == 1:
                        data_set_local[i + 1] = np.array(yhat_encoded)
                    else:
                        data_set_local[i + 1] = np.append(data_set_local[i][-(voice.win_size-1)*voice.vec_size:], np.array(yhat_encoded))
            loss = log_loss(y, yhat[1])
            losses.append(loss)
        losses_all_weights.append(sum(losses)/len(losses))
    if len(losses_all_weights) > 1:
        return losses_all_weights
    return losses_all_weights[0]


def generate_notes(voice_note: object, voice_octave: object, n: int) -> Callable:
    voice_note_copy = copy.deepcopy(voice_note)
    voice_octave_copy = copy.deepcopy(voice_octave)
    generated_notes = []
    yhat_notes = []
    yhat_octaves = []
    for i in range(n):
        yhat_note = predict_note(voice_note_copy, -1, dataset_type="all")[0]
        yhat_octave = predict_note(voice_octave_copy, -1, dataset_type="all")[0]

        while yhat_note == -1 and yhat_octave != -2:
            yhat_note = random.choice(voice_note_copy.set_reduced_notes)

        while yhat_octave == -2 and yhat_note != -1:
            yhat_octave = random.choice(voice_octave_copy.set_reduced_notes)

        yhat_notes.append(yhat_note)
        yhat_octaves.append(yhat_octave)

        yhat_note_encoded = np.array([one_hot_encode(yhat_note, voice_note_copy.set_reduced_notes)])
        yhat_octave_encoded = np.array([one_hot_encode(yhat_octave, voice_octave_copy.set_reduced_notes)])

        yhat_note_encoded_windowed = np.array([
            np.append(
            voice_note_copy.X_encoded_windowed[-1][-(voice_note_copy.win_size - 1) * voice_note_copy.vec_size:],
            np.array(yhat_note_encoded))
        ])
        yhat_octave_encoded_windowed = np.array([
            np.append(
            voice_octave_copy.X_encoded_windowed[-1][-(voice_octave_copy.win_size - 1) * voice_octave_copy.vec_size:],
            np.array(yhat_octave_encoded)
            )
        ])
        voice_note_copy.X_encoded_windowed = np.concatenate((voice_note_copy.X_encoded_windowed,
                                                             yhat_note_encoded_windowed))
        voice_octave_copy.X_encoded_windowed = np.concatenate((voice_octave_copy.X_encoded_windowed,
                                                               yhat_octave_encoded_windowed))
        yhat_midi = yhat_note + (yhat_octave+1)*12
        if yhat_midi == -13:  # -13 can happen only if note=-1 and octave=-2, if note!=-1 then octave is reiterated
            yhat_midi = 0
        generated_notes.append(yhat_midi)
    return generated_notes


def write_to_txt(pred_notes, filename):
    pred_notes = pd.DataFrame([preds for preds in pred_notes]).transpose()
    np.savetxt(filename, pred_notes, fmt="%s")


# 0) Read data "model\\F.txt"
my_path = os.path.abspath('.')
data = pd.read_csv(os.path.join(my_path, "F.txt"), sep="\t", header=None)
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


if __name__ == "__main__":
    p = Pool(8)
    p1 = p.apipe(create_save_pickle('note', 0))
    p2 = p.apipe(create_save_pickle('note', 1))
    p3 = p.apipe(create_save_pickle('note', 2))
    p4 = p.apipe(create_save_pickle('note', 3))
    p5 = p.apipe(create_save_pickle('octave', 0))
    p6 = p.apipe(create_save_pickle('octave', 1))
    p7 = p.apipe(create_save_pickle('octave', 2))
    p8 = p.apipe(create_save_pickle('octave', 3))
    p1.get()
    p2.get()
    p3.get()
    p4.get()
    p5.get()
    p6.get()
    p7.get()
    p8.get()

