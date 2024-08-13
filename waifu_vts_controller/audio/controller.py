import asyncio
import time
from typing import Dict, Literal, Union
import librosa
import numpy as np
from fastdtw import fastdtw
import pyvts
from scipy.spatial.distance import euclidean
from waifu_vts_controller.audio.utils import AudioPlayer

class AudioProcessor:
    def __init__(self, sample_rate: int = None, n_mfcc: int = 13, window_size: float = 0.25, hop_length: float = 0.125):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.hop_length = hop_length

    def compute_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        # Normalize the audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        window_size_samples = int(self.window_size * self.sample_rate)
        hop_length_samples = int(self.hop_length * self.sample_rate)

        # Compute the MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=window_size_samples,
            hop_length=hop_length_samples,
        )
        return mfcc

    def compute_mfcc_from_file(self, file_path: str) -> np.ndarray:
        # Load the audio file
        audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
        self.sample_rate = sr if self.sample_rate is None else self.sample_rate

        # Compute the MFCC
        return self.compute_mfcc(audio_data)

    def pad_or_truncate(self, mfcc: np.ndarray, target_shape: tuple) -> np.ndarray:
        # Pad or truncate MFCC matrix to match the target shape
        if mfcc.shape[1] < target_shape[1]:
            # Pad with zeros if shorter
            padding = target_shape[1] - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), "constant")
        elif mfcc.shape[1] > target_shape[1]:
            # Truncate if longer
            mfcc = mfcc[:, : target_shape[1]]
        return mfcc

    def classify_phoneme(
        self, 
        mfcc_phonemes: Dict[Literal["a", "i", "u", "e", "o", "n"], np.ndarray], 
        mfcc_to_classify: np.ndarray
    ) -> Literal["a", "i", "u", "e", "o", "n"]:
        # Determine the maximum shape to compare
        target_shape = max(
            (mfcc.shape for mfcc in mfcc_phonemes.values()), key=lambda x: x[1]
        )

        # Pad or truncate the MFCC to classify to match the target shape
        mfcc_to_classify = self.pad_or_truncate(mfcc_to_classify, target_shape)

        # Compare the MFCC of the phoneme to classify with the MFCCs of the known phonemes using DTW
        phoneme_distances: Dict[Literal["a", "i", "u", "e", "o", "n"], float] = {}
        for phoneme, known_mfcc in mfcc_phonemes.items():
            known_mfcc = self.pad_or_truncate(known_mfcc, target_shape)
            distance, _ = fastdtw(mfcc_to_classify.T, known_mfcc.T, dist=euclidean)
            phoneme_distances[phoneme] = distance

        # Return the phoneme with the smallest DTW distance
        return min(phoneme_distances, key=phoneme_distances.get)

    def amplify_calculation(self, audio_chunk: np.ndarray) -> float:
        # Calculate amplitude based on the audio chunk
        amplitude = np.max(np.abs(audio_chunk))
        amplitude = float(amplitude) + 0.5
        # Max amplitude is 1.0
        amplitude = min(amplitude, 1.0)
        amplitude = max(amplitude, 0.0)
        return float(amplitude)
       
# Control
class VTSAudioController:
    def __init__(self, vts: pyvts.vts, audio_processor: AudioProcessor):
        self.vts = vts
        self.audio_processor = audio_processor

    async def connect(self):
        if (self.vts.get_authentic_status()) != 2:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            return await self.vts.request_authenticate()
        return True

    async def set_mouth_parameters(self):
        await self.vts.request(
            self.vts.vts_request.requestCustomParameter(
                parameter="WaifuMouthX",
                min=0,
                max=1,
                default_value=0,
                info="X factor of the mouth",
            )
        )

        await self.vts.request(
            self.vts.vts_request.requestCustomParameter(
                parameter="WaifuMouthY",
                min=0,
                max=1,
                default_value=0.0,
                info="Y factor of the mouth",
            )
        )

    async def update_mouth_based_on_phonemes(
        self,
        phoneme: Literal["a", "i", "u", "e", "o", "n"],
        amp_factor: float,
        mouth_factor={
            "a": {"x": 1.0, "y": 0.6},
            "i": {"x": 1.0, "y": 0.1},
            "u": {"x": 0.2, "y": 1.0},
            "e": {"x": 1.0, "y": 0.4},
            "o": {"x": 0.35, "y": 1.0},
            "n": {"x": 0.0, "y": 0.0},
        }
    ):
        await self.vts.request(
            self.vts.vts_request.requestSetParameterValue(
                parameter="WaifuMouthX", value=mouth_factor[phoneme]["x"] * amp_factor
            )
        )

        await self.vts.request(
            self.vts.vts_request.requestSetParameterValue(
                parameter="WaifuMouthY", value=mouth_factor[phoneme]["y"] * amp_factor
            )
        )

    async def close(self):
        await self.vts.close()

    async def play_audio_with_mouth_movement(
        self,
        audio_path: Union[str, np.ndarray],
        phoneme_files: Dict[str, str]
    ):
        # Connect and set mouth parameters
        await self.connect()
        await self.set_mouth_parameters()

        # Compute MFCC for each phoneme
        phonemes_mfcc = {
            phoneme: self.audio_processor.compute_mfcc_from_file(path) 
            for phoneme, path in phoneme_files.items()
        }

        # Load audio file or use provided audio data
        if isinstance(audio_path, str):
            audio_data, sr = librosa.load(audio_path, sr=None)
        else:
            audio_data = audio_path
            sr = self.audio_processor.sample_rate

        self.audio_processor.sample_rate = int(sr)

        # Define window and hop length in samples
        window_size_samples = int(self.audio_processor.window_size * sr)
        hop_length_samples = int(self.audio_processor.hop_length * sr)

        time_stamp = 0.0
        last_phoneme = None
        counter = 0

        # Play Audio in background
        asyncio.get_event_loop().run_in_executor(
            None, AudioPlayer.play_audio_chunk, audio_data, sr
        )
        
        # Process audio data in chunks
        for i in range(0, len(audio_data), hop_length_samples):
            wait_for = hop_length_samples / sr

            if len(audio_data) - i < window_size_samples:
                print("End of audio data")
                break

            segment = audio_data[i: i + window_size_samples]
            mfcc = self.audio_processor.compute_mfcc(segment)
            classified_phoneme = self.audio_processor.classify_phoneme(phonemes_mfcc, mfcc)

            if classified_phoneme != last_phoneme:
                print(f"Time: {time_stamp} - Phoneme: {classified_phoneme}")
                last_phoneme = classified_phoneme

            time_stamp += hop_length_samples / sr
            counter += 1
            if counter % 100 == 0:
                print(f"Time: {time_stamp} - Phoneme: {classified_phoneme}")

            # Update mouth factor
            amp = self.audio_processor.amplify_calculation(segment)
            print(amp)
            await self.update_mouth_based_on_phonemes(classified_phoneme, amp_factor=amp)

            # Wait for the audio chunk to finish playing
            time.sleep(wait_for * 0.75)

        await self.close()
