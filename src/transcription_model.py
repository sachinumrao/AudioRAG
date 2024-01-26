import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from transformers import modelcard

DEBUG = os.environ.get("DEBUG", "0")


class ASRModel(ABC):
    @abstractmethod
    def get_transcript(self, audio_filepath: str) -> str:
        pass


class HuggingASRModel(ASRModel):
    """
    ASR Model using hugginghface ASR pipeline with whisper model.
    """

    def __init__(self, model_size="tiny"):
        from transformers import pipeline

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.batch_size = 1

        self.model_size = model_size

        print(f"Downloading whisper {model_size} model...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}.en",
            chunk_length_s=15,
            device=self.device,
        )
        print("Model download complete...")

    def get_transcript(self, audio_filepath: str) -> str:
        """Transcribe audio file into english"""
        print(f"Transcribing file: {audio_filepath}")
        t1 = time.perf_counter()
        result = self.pipe(audio_filepath)
        t2 = time.perf_counter()
        print(f"Time Taken in transcription: {t2-t1}s")
        return result["text"]


class WhisperASRModel(ASRModel):
    """
    ASR Model using the whisper librayr from openai. It uses v1 of the whisper model series.
    """

    def __init__(self, model_size="tiny"):
        import whisper

        self.model_size = model_size
        self.model = whisper.load_model(self.model_size)

        pass

    def get_transcript(self, audio_filepath: str) -> str:
        """Transcribe audio file into english"""

        print(f"Transcribing file: {audio_filepath}")
        t1 = time.perf_counter()
        result = self.model.transcribe(str(audio_filepath))

        t2 = time.perf_counter()
        print(f"Time Taken in Transcription: {t2-t1}s")
        return result["text"]


def WhisperCPPASRModel(ASRModel):
    """
    ASR Model using the whisper-cpp python lib.
    """

    def __init__(self, model_path: str):
        from whisper_cpp_python import Whisper

        self.model = Whisper(model_path=model_path)

    def get_transcript(self, audio_filepath: str) -> str:
        """Transcribe audio file into english"""

        t1 = time.perf_counter()
        result = self.model.transcribe(open(str(audio_filepath)), language="en")
        t2 = time.perf_counter()
        print(f"Time Taken in Transcription: {t2-t1}s")

        return result["text"])


def main():
    pass




if __name__ == "__main__":
    # openai_whipser_trnasription() # working fine. for 42.2mins tiny taskes 207sec, small takes 375sec
    # test_faster_whisper() # hanging
    # test_whisper_cpp() # library not loading, missing shared library
    main()  # working tiny model takes 260sec

## TODO:
# 1. Fix bugs in whisperc-cpp inference
