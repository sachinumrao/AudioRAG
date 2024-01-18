import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEBUG = os.environ.get("DEBUG", "0")


class AudioModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.batch_size = 1

        model_size = "tiny"

        print("Downloading whisper model...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}.en",
            chunk_length_s=15,
            device=self.device,
        )
        print("Model download complete...")

    def get_transcript(self, audio_filepath):
        print(f"Transcribing file: {audio_filepath}")
        result = self.pipe(audio_filepath)
        return result["text"]


def main():
    audio_directory = "./../audio/"
    audio_filename = "State_of_GPT_|_BRK216HFS.mp3"
    audio_filepath = Path(audio_directory, audio_filename)

    t1 = time.perf_counter()
    model = AudioModel()
    t2 = time.perf_counter()

    print("Time taken to load the model: ", t2 - t1)

    t3 = time.perf_counter()
    transcript = model.get_transcript(str(audio_filepath))
    t4 = time.perf_counter()

    print("Time taken for transcription: ", t4 - t3)

    print("Words in transcript: ", len(transcript.split()))

    # output_path = Path("./../transcripts/", "karpathy_state_of_gpt.txt")

    # with open(output_path, "w+") as f:
    #    f.writelines(transcript)


def test_faster_whisper():
    from faster_whisper import WhisperModel

    audio_directory = "./../audio/"
    audio_filename = "State_of_GPT_|_BRK216HFS.mp3"
    audio_filepath = Path(audio_directory, audio_filename)
    model_size = "tiny.en"  # "large-v3"

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8

    if DEBUG == "1":
        print("Loading the model...")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(str(audio_filepath), beam_size=1)

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    if DEBUG == "1":
        print("Starting Transcription...")
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def openai_whipser_trnasription():
    import whisper

    audio_directory = "./../audio/"
    audio_filename = "State_of_GPT_|_BRK216HFS.mp3"
    audio_filepath = Path(audio_directory, audio_filename)

    model_type = "base"

    t1 = time.perf_counter()
    model = whisper.load_model(model_type)
    t2 = time.perf_counter()
    print("Time Taken in Loading Model: ", t2 - t1)

    t3 = time.perf_counter()
    result = model.transcribe(str(audio_filepath))
    t4 = time.perf_counter()
    print("Time Taken in Transcription: ", t4 - t3)

    with open(
        f"./../transcripts/{audio_filename.split('.')[0]}_{model_type}_transcription.txt",
        "w",
        encoding="utf-8",
    ) as txt:
        txt.write(result["text"])


def test_whisper_cpp():
    from whisper_cpp_python import Whisper

    audio_directory = "./../audio/"
    audio_filename = "State_of_GPT_|_BRK216HFS.mp3"
    audio_filepath = Path(audio_directory, audio_filename)

    model_type = "tiny.en"

    t1 = time.perf_counter()
    model = Whisper(model_path="./models/ggml-tiny.bin")
    t2 = time.perf_counter()
    print("Time Taken in Loading Model: ", t2 - t1)

    t3 = time.perf_counter()
    result = model.transcribe(open(str(audio_filepath)), language="en")
    t4 = time.perf_counter()
    print("Time Taken in Transcription: ", t4 - t3)

    print(result["text"])


if __name__ == "__main__":
    # openai_whipser_trnasription() # working fine. for 42.2mins tiny taskes 207sec, small takes 375sec
    # test_faster_whisper() # hanging
    # test_whisper_cpp() # library not loading, missing shared library
    main()  # working tiny model takes 260sec

## TODO:
# 1. Refactor AudioModel to generic ASRModel class which supports all 4 backends with optional gpu support
# 2. Build model classes for all backends
# 3. Fiox bugs in faster whisper inference
# 4. Fix bugs in whisperc-cpp inference
