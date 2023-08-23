import librosa


def read_audio(filepath, sampling_rate):
    data, sr = librosa.load(filepath, sr=sampling_rate)

    # Prepare input for the model
    speech = {
        "path": filepath,
        "array": data,
        "sampling_rate": sr,
    }
    return speech
