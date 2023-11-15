from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class AudioModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.batch_size = 1

        model_id = "distil-whisper/distil-medium.en"
        
        print("Downloading whisper model...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        
        print("Model download complete...")
        self.model.to(self.device)
        
        self.model = self.model.to_bettertransformer()

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=self.batch_size,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def get_transcript(self, audio_filepath):
        print(f"Transcribing file: {audio_filepath}")
        result = self.pipe(audio_filepath)
        return result
    
    
    
def main():
    
    audio_directory = "./../audio/"
    audio_filename = "State_of_GPT_|_BRK216HFS.mp3"
    audio_filepath = Path(audio_directory, audio_filename)
    
    model = AudioModel()
    transcript = model.get_transcript(str(audio_filepath))
    print("Transcript:")
    print(transcript)


if __name__ == "__main__":
    main()