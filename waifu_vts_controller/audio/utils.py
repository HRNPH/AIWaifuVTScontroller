import numpy as np
import soundcard as sc

class AudioPlayer:
    @staticmethod
    def play_audio_chunk(chunk: np.ndarray, sample_rate: int):
        # Play audio from a NumPy array using soundcard
        default_speaker = sc.default_speaker()
        with default_speaker.player(samplerate=sample_rate) as player:
            player.play(chunk)
     