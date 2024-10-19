import asyncio
import time
from typing import Any, Dict, Literal, Union
import librosa
import numpy as np
from fastdtw import fastdtw
import pyvts
from scipy.spatial.distance import euclidean
from waifu_vts_controller.audio.utils import AudioPlayer

# audio
class AudioProcessor:
    def __init__(self, sample_rate: int = None, n_mfcc: int = 13, window_size: float = 0.25, hop_length: float = 0.125):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.hop_length = hop_length
        
    def compute_mfcc(self, audio_data: np.ndarray) -> Union[np.ndarray, None]:
        """Compute the Mel-frequency cepstral coefficients (MFCC) of the audio data, with additional chroma and spectral contrast features."""
        if not np.isfinite(audio_data).all():
            raise ValueError("Audio buffer contains invalid values (NaN or infinity).")
        
        if not np.any(audio_data):
            return None

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        else:
            raise ValueError("Audio data has zero maximum value, cannot normalize.")

        # Apply noise reduction using preemphasis
        audio_data = librosa.effects.preemphasis(audio_data)

        window_size_samples = int(self.window_size * self.sample_rate)
        hop_length_samples = int(self.hop_length * self.sample_rate)

        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=window_size_samples,
            hop_length=hop_length_samples,
        )

        # Compute additional features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate, hop_length=hop_length_samples)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate, hop_length=hop_length_samples)

        # Normalize MFCC
        mfcc = self.normalize_mfcc(mfcc)

        # Pad or truncate chroma and spectral contrast to match the MFCC shape
        chroma = self.pad_or_truncate(chroma, mfcc.shape)
        spectral_contrast = self.pad_or_truncate(spectral_contrast, mfcc.shape)

        # Concatenate MFCC with chroma and spectral contrast to enrich the feature set
        features = np.vstack([mfcc, chroma, spectral_contrast])
        
        return features

    def compute_mfcc_from_file(self, file_path: str) -> Union[np.ndarray, None]:
        audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
        self.sample_rate = sr if self.sample_rate is None else self.sample_rate
        return self.compute_mfcc(audio_data)

    def normalize_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """Normalize MFCC values to have mean 0 and standard deviation 1."""
        return (mfcc - np.mean(mfcc)) / np.std(mfcc)

    def pad_or_truncate(self, mfcc: np.ndarray, target_shape: tuple) -> np.ndarray:
        if mfcc.shape[1] < target_shape[1]:
            padding = target_shape[1] - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), "constant")
        elif mfcc.shape[1] > target_shape[1]:
            mfcc = mfcc[:, : target_shape[1]]
        return mfcc

    def classify_phoneme(
        self, 
        mfcc_phonemes: Dict[Literal["a", "i", "u", "e", "o", "n"], np.ndarray], 
        mfcc_to_classify: Union[np.ndarray, None]
    ) -> Literal["a", "i", "u", "e", "o", "n"]:
        if mfcc_to_classify is None:
            return "n"

        # Determine the maximum shape to compare
        target_shape = max(
            (mfcc.shape for mfcc in mfcc_phonemes.values()), key=lambda x: x[1]
        )

        # Pad or truncate the MFCC to classify to match the target shape
        mfcc_to_classify = self.pad_or_truncate(mfcc_to_classify, target_shape)

        # Compare the MFCC using DTW and log distances
        phoneme_distances: Dict[Literal["a", "i", "u", "e", "o", "n"], float] = {}
        for phoneme, known_mfcc in mfcc_phonemes.items():
            known_mfcc = self.pad_or_truncate(known_mfcc, target_shape)
            distance, _ = fastdtw(mfcc_to_classify.T, known_mfcc.T, dist=euclidean)
            phoneme_distances[phoneme] = distance

        return min(phoneme_distances, key=phoneme_distances.get)

    def amplify_calculation(self, audio_chunk: np.ndarray) -> float:
        amplitude = np.max(np.abs(audio_chunk))
        amplitude = (float(amplitude) + 0.5) * 1 + np.mean(np.abs(audio_chunk)) * 0.5
        amplitude = min(amplitude, 1.0)
        amplitude = max(amplitude, 0.0)
        return float(amplitude)

# Control
class VTSAudioController:
    def __init__(self, vts: pyvts.vts, audio_processor: AudioProcessor):
        self.vts = vts
        self.audio_processor = audio_processor
        self.phoneme_queue: asyncio.Queue = asyncio.Queue()

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
        phoneme_files: Dict[Literal["a", "i", "u", "e", "o", "n"], str]
    ):
        await self.connect()
        await self.set_mouth_parameters()

        # Precompute the MFCC for each phoneme
        phonemes_mfcc = {
            phoneme: self.audio_processor.compute_mfcc_from_file(path) 
            for phoneme, path in phoneme_files.items()
        }

        # Load the audio data
        if isinstance(audio_path, str):
            audio_data, sr = librosa.load(audio_path, sr=None)
        else:
            audio_data = audio_path
            sr = self.audio_processor.sample_rate

        sr = int(sr)
        self.audio_processor.sample_rate = sr

        # Calculate the window size and hop length in samples
        window_size_samples = int(self.audio_processor.window_size * sr)
        hop_length_samples = int(self.audio_processor.hop_length * sr)

        # Calculate the sleep time based on the hop length and sample rate
        sleep_time = hop_length_samples / sr

        # Precompute all phoneme chunks before starting playback
        print(f"Precomputing all phoneme chunks...")

        phoneme_data = []  # Store all phoneme chunks for later consumption

        for i in range(0, len(audio_data), hop_length_samples):
            if len(audio_data) - i < window_size_samples:
                print("End of audio data during precomputation")
                break

            segment = audio_data[i:i + window_size_samples]
            mfcc = self.audio_processor.compute_mfcc(segment)
            classified_phoneme = self.audio_processor.classify_phoneme(phonemes_mfcc, mfcc)
            phoneme_data.append((classified_phoneme, segment))

        print(f"Precomputed {len(phoneme_data)} phoneme chunks")

        # Start audio playback after precomputing all phonemes
        print("Starting playback...")
        asyncio.get_event_loop().run_in_executor(None, AudioPlayer.play_audio_chunk, audio_data, sr)

        # Sync the precomputed phonemes with the real-time audio playback
        start_time = time.time()

        for i, (classified_phoneme, segment) in enumerate(phoneme_data):
            expected_time = i * sleep_time
            current_time = time.time() - start_time

            if current_time < expected_time:
                await asyncio.sleep(expected_time - current_time)

            amp = self.audio_processor.amplify_calculation(segment)
            await self.update_mouth_based_on_phonemes(classified_phoneme, amp_factor=amp)

        print(f"LipSync End At: {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
        await self.close()