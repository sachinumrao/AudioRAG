import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class AudioModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.batch_size = 1

        print("Downloading whisper model...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
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

    output_path = Path("./../transcripts/", "karpathy_state_of_gpt.txt")

    with open(output_path, "w+") as f:
        f.writelines(transcript)


if __name__ == "__main__":
    main()
