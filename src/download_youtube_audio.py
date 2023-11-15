from pathlib import Path

from pytube import YouTube


def get_audio_from_youtube(url, save_directory):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    
    cln_title = str(yt.title).replace(" ", "_") + ".mp3"
    audio.download(output_path=save_directory, filename=cln_title)
    pass

def main():
    url = "https://www.youtube.com/watch?v=bZQun8Y4L2A"
    destination = "./../audio/"
    get_audio_from_youtube(url, destination)
    
if __name__ == "__main__":
    main()