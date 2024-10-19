import numpy as np
import soundcard as sc
import time

class AudioPlayer:
    @staticmethod
    def play_audio_chunk(chunk: np.ndarray, sample_rate: int):
        # Play audio from a NumPy array using soundcard
        default_speaker = sc.default_speaker()
        audio_duration = chunk.shape[0] / sample_rate  # Calculate the duration in seconds

        start_time = time.time()  # Record the start time
        with default_speaker.player(samplerate=sample_rate) as player:
            player.play(chunk)

        end_time = start_time + audio_duration  # Calculate the end time
        print(f"Audio playback end at: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
