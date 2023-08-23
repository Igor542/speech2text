import speech2text
from speech2text.model import Whisper as Model
from speech2text.utils import read_audio


m = Model()


samples = [
    "example1.ogx",
    "example2.ogx",
    "example3.ogx",
]


for sample in samples:
    speech = read_audio(sample, Model.sampling_rate)

    transcription = m.process(speech)
    print(transcription)
