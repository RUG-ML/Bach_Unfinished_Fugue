import matplotlib.pyplot as plt
from model.bash_baseline_v2 import notes_transformed_to_octaves, set_notes_transformed_to_octaves

def octave_info():
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