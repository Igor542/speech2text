from transformers import WhisperProcessor, WhisperForConditionalGeneration


class Whisper:
    sampling_rate = 16000

    def __init__(self, model_name="openai/whisper-medium", lang="russian"):
        self.model_name = model_name
        self.lang = lang
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=lang, task="transcribe"
        )

    def process(self, data):
        input_features = self.processor(
            data["array"], sampling_rate=data["sampling_rate"], return_tensors="pt"
        ).input_features

        # generate token ids
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=self.forced_decoder_ids, max_length=448
        )

        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription
