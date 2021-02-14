from model import Voice
from model import fit_model_to_voice
from model import generate_notes
from model import write_to_txt

if __name__ == "__main__":
    v1_window = 160
    v2_window = 160
    v3_window = 160
    v4_window = 180
    note_v1 = fit_model_to_voice(Voice('note', 0, v1_window), regularization=True)
    octave_v1 = fit_model_to_voice(Voice('octave', 0, v1_window), regularization=True)
    print("First voice complete.")

    note_v2 = fit_model_to_voice(Voice('note', 1, v2_window), regularization=True)
    octave_v2 = fit_model_to_voice(Voice('octave', 1, v2_window), regularization=True)
    print("Second voice complete.")

    note_v3 = fit_model_to_voice(Voice('note', 2, v3_window), regularization=True)
    octave_v3 = fit_model_to_voice(Voice('octave', 2, v3_window), regularization=True)
    print("Third voice complete.")

    note_v4 = fit_model_to_voice(Voice('note', 3, v4_window), regularization=True)
    octave_v4 = fit_model_to_voice(Voice('octave', 3, v4_window), regularization=True)
    print("Fourth voice complete.")

    pred550 = [generate_notes(note_v1, octave_v1, 550),
               generate_notes(note_v2, octave_v2, 550),
               generate_notes(note_v3, octave_v3, 550),
               generate_notes(note_v4, octave_v4, 550)]

    write_to_txt(pred550, "pred550.txt")



