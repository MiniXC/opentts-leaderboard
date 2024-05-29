from abc import ABC, abstractmethod
import os
import tempfile
import re

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, AutoTokenizer, Wav2Vec2ForCTC
from parler_tts import ParlerTTSForConditionalGeneration
from pathlib import Path
from tqdm.auto import tqdm
from pyannote.audio import Model
from dotenv import load_dotenv
from pyannote.audio import Inference
from datasets import load_dataset
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from miipher.lightning_module import MiipherLightningModule
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
from voicefixer import VoiceFixer
import whisper
from jiwer import wer
import hydra
from masked_prosody_model import MaskedProsodyModel
from speech_collator.measures import (
    PitchMeasure,
    EnergyMeasure,
    VoiceActivityMeasure,
)

from generate_data import (
    generate_data,
    N_SAMPLES
)
from frechet_distance import frechet_distance
from dvector.wav2mel import Wav2Mel

load_dotenv()


class EmbeddingModel(ABC):
    model_layer = 6

    def process_audio(self, audio, noise=False):
        if isinstance(audio, str) or isinstance(audio, Path):
            audio, sr = torchaudio.load(audio)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            sr = 16000
        if isinstance(audio, tuple):
            audio, sr = audio
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float().unsqueeze(0)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            sr = 16000
        if noise:
            audio = torch.randn_like(audio)
            audio = audio / audio.abs().max()
        return audio, sr

    def get_embedding(self, audio, text, speaker, noise=False):
        audio, sr = self.process_audio(audio, noise=noise)
        return self._get_embedding(audio, sr, text, speaker)

    def get_mu_sigma(self, ds_name):
        self._dataset_hook(ds_name)
        noise = False
        prefix = self.__class__.__name__.lower()
        Path("cache").mkdir(exist_ok=True, parents=True)
        if (Path("cache") / f"{prefix}_{ds_name}_mu.npy").exists():
            mu = np.load(Path("cache") / f"{prefix}_{ds_name}_mu.npy")
            sigma = np.load(Path("cache") / f"{prefix}_{ds_name}_sigma.npy")
            return mu, sigma
        features = []
        audios, texts, speakers = self.get_audios(ds_name)
        for audio, text, speaker in tqdm(zip(audios, texts, speakers), f"generating {ds_name}"):
            features.append(self.get_embedding(audio, text, speaker, noise=noise))
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        np.save(Path("cache") / f"{prefix}_{ds_name}_mu.npy", mu)
        np.save(Path("cache") / f"{prefix}_{ds_name}_sigma.npy", sigma)
        return mu, sigma

    def get_noise_mu_sigma(self):
        self._dataset_hook("reference.test")
        noise = True
        ds_name = "noise"
        prefix = self.__class__.__name__.lower()
        Path("cache").mkdir(exist_ok=True, parents=True)
        if (Path("cache") / f"{prefix}_{ds_name}_mu.npy").exists():
            mu = np.load(Path("cache") / f"{prefix}_{ds_name}_mu.npy")
            sigma = np.load(Path("cache") / f"{prefix}_{ds_name}_sigma.npy")
            return mu, sigma
        features = []
        audios, texts, speakers = self.get_audios("reference.test")
        for audio, text, speaker in tqdm(zip(audios, texts, speakers), f"generating {ds_name}"):
            features.append(self.get_embedding(audio, text, speaker, noise=noise))
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        np.save(Path("cache") / f"{prefix}_{ds_name}_mu.npy", mu)
        np.save(Path("cache") / f"{prefix}_{ds_name}_sigma.npy", sigma)
        return mu, sigma

    def get_audios(self, ds_name):
        return generate_data(ds_name, self.device)

    def get_frechet(self, ds_name):
        val = frechet_distance(
            *self.get_mu_sigma(ds_name),
            *self.get_mu_sigma(f"reference.test")
        )
        worst_value = frechet_distance(
            *self.get_mu_sigma("reference.test"),
            *self.get_noise_mu_sigma()
        )
        best_value = frechet_distance(
            *self.get_mu_sigma("reference.test"),
            *self.get_mu_sigma("reference.dev"),
        )
        # convert to score from 0 to 1
        result = 1 - ((val - best_value) / (worst_value - best_value))
        return np.round(result * 100, 2)
    

    @abstractmethod
    def _get_embedding(self, audio, sr, text):
        pass

    def _dataset_hook(self, ds):
        pass


class MFCC(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.device = device
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
        )

    def _get_embedding(self, audio, sr, text, speaker=None):
        mfcc = self.mfcc(audio)
        mfcc = mfcc.mean(2).detach().cpu().numpy()[0]
        return mfcc


class Hubert(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_model = hubert_model.to(device)
        self.hubert_model.eval()
        self.device = device

    def _get_embedding(self, audio, sr, text, speaker=None):
        input_values = self.hubert_processor(
            audio, return_tensors="pt", sampling_rate=16000
        ).input_values
        with torch.no_grad():
            features = self.hubert_model(input_values[0].to(self.device), output_hidden_states=True).hidden_states
        features = features[self.model_layer].mean(1).detach().cpu().numpy()[0]
        return features


class Wav2Vec2(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = wav2vec2_model.to(device)
        self.wav2vec2_model.eval()
        self.device = device

    def _get_embedding(self, audio, sr, text, speaker=None):
        input_values = self.wav2vec2_processor(
            audio, return_tensors="pt", sampling_rate=16000
        ).input_values
        with torch.no_grad():
            features = self.wav2vec2_model(input_values[0].to(self.device), output_hidden_states=True).hidden_states
        features = features[self.model_layer].mean(1).detach().cpu().numpy()[0]
        return features

class DVector(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.wav2mel = Wav2Mel()
        self.dvector = torch.jit.load("dvector/dvector.pt").eval()
        self.dvector = self.dvector.to(device)
        self.device = device

    def _get_embedding(self, audio, sr, text, speaker=None):
        mel_tensor = self.wav2mel(audio, sr)  # shape: (frames, mel_dim)
        mel_tensor = mel_tensor.to(self.device)
        emb_tensor = self.dvector.embed_utterance(mel_tensor)
        return emb_tensor.detach().cpu().numpy()[0]

class DVectorIntra(EmbeddingModel):
    def __init__(self, dataset, device="cuda"):
        self.wav2mel = Wav2Mel()
        self.dvector = torch.jit.load("dvector/dvector.pt").eval()
        self.dvector = self.dvector.to(device)
        self.device = device
    
    def _dataset_hook(self, ds):
        self.speaker_centroids = self.calculate_speaker_centroids(ds)
        
    def calculate_speaker_centroids(self, dataset):
        speaker_centroids = {}
        audios, texts, speakers = generate_data(dataset)
        for audio, text, speaker in zip(audios, texts, speakers):
            if speaker not in speaker_centroids:
                speaker_centroids[speaker] = []
            audio, sr = self.process_audio(audio)
            mel_tensor = self.wav2mel(audio, sr)
            mel_tensor = mel_tensor.to(self.device)
            emb_tensor = self.dvector.embed_utterance(mel_tensor)
            speaker_centroids[speaker].append(emb_tensor.detach().cpu().numpy()[0])
        for speaker in speaker_centroids:
            speaker_centroids[speaker] = np.mean(speaker_centroids[speaker], axis=0)
        return speaker_centroids

    def _get_embedding(self, audio, sr, text, speaker):
        mel_tensor = self.wav2mel(audio, sr)  # shape: (frames, mel_dim)
        mel_tensor = mel_tensor.to(self.device)
        emb_tensor = self.dvector.embed_utterance(mel_tensor)
        emb_tensor = emb_tensor.detach().cpu().numpy()[0]
        return emb_tensor - self.speaker_centroids[speaker]
        

class XVector(EmbeddingModel):
    def __init__(self, device="cuda"):
        xvector_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        xvector_model.to(device)
        self.xvector = Inference(xvector_model, window="whole")
        self.device = device

    def _get_embedding(self, audio, sr, text, speaker=None):
        # save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, audio, sr)
            # get embedding
            try:
                embedding = self.xvector(f.name)
            except RuntimeError:
                embedding = np.zeros(512)
        return embedding[0]

class XVectorIntra(EmbeddingModel):
    def __init__(self, dataset, device="cuda"):
        xvector_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        xvector_model.to(device)
        self.xvector = Inference(xvector_model, window="whole")
        self.device = device
        self.speaker_centroids = self.calculate_speaker_centroids(dataset)

    def _dataset_hook(self, ds):
        self.speaker_centroids = self.calculate_speaker_centroids(ds)

    def calculate_speaker_centroids(self, dataset):
        speaker_centroids = {}
        audios, texts, speakers = generate_data(dataset)
        for audio, text, speaker in zip(audios, texts, speakers):
            if speaker not in speaker_centroids:
                speaker_centroids[speaker] = []
            # save to temp file
            audio, sr = self.process_audio(audio)
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                torchaudio.save(f.name, audio, sr)
                # get embedding
                try:
                    embedding = self.xvector(f.name)
                except RuntimeError:
                    embedding = np.zeros(512)
                speaker_centroids[speaker].append(embedding[0])
        for speaker in speaker_centroids:
            speaker_centroids[speaker] = np.mean(speaker_centroids[speaker], axis=0)
        return speaker_centroids

    def _get_embedding(self, audio, sr, text, speaker):
        # save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, audio, sr)
            # get embedding
            try:
                embedding = self.xvector(f.name)
            except RuntimeError:
                embedding = np.zeros(512)
        return embedding[0] - self.speaker_centroids[speaker]
    
class Miipher(EmbeddingModel):
    """
    See: https://github.com/Wataru-Nakata/miipher/blob/main/examples/demo.py
    """

    def __init__(self, device="cuda", audio_features="mfcc", audio_features_device="cuda"):
        miipher_path = "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/miipher.ckpt"
        self.miipher = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location=device)
        self.preprocessor = PreprocessForInfer(self.miipher.cfg)
        self.vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint(
            "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/vocoder_finetuned.ckpt",
            map_location=device
        )
        self.xvector_model = hydra.utils.instantiate(self.vocoder.cfg.data.xvector.model)
        self.xvector_model = self.xvector_model.to("cpu")
        self.device = device
        self.audio_features_type = audio_features
        if audio_features == "mfcc":
            self.audio_features = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
            )
        elif audio_features == "hubert":
            self.audio_features = Hubert(device=audio_features_device)
        elif audio_features == "wav2vec2":
            self.audio_features = Wav2Vec2(device=audio_features_device)

    def _get_embedding(self, audio, sr, text, speaker=None):
        wav = audio
        batch = self.preprocessor.process(
            'test',
            (torch.tensor(wav),sr),
            word_segmented_text=text,
            lang_code="eng-us"
        )
        self.miipher.feature_extractor(batch)
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            _,
        ) = self.miipher.feature_extractor(batch)
        phone_feature = phone_feature.to(self.device)
        speaker_feature = speaker_feature.to(self.device)
        degraded_ssl_feature = degraded_ssl_feature.to(self.device)
        with torch.no_grad():
            cleaned_ssl_feature, _ = self.miipher(phone_feature, speaker_feature, degraded_ssl_feature)
            degraded_wav_16k = batch['degraded_wav_16k'].view(1,-1)
            vocoder_xvector = self.xvector_model.encode_batch(degraded_wav_16k.cpu()).squeeze(1).to(self.device)
            cleaned_wav = self.vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector})[0].T.view(1,-1)
            cleaned_wav = cleaned_wav / cleaned_wav.abs().max()
            original_wav = self.vocoder.generator_forward({"input_feature": degraded_ssl_feature, "xvector": vocoder_xvector})[0].T.view(1,-1)
            original_wav = original_wav / original_wav.abs().max()
        # get difference between original and cleaned
        if hasattr(self, "audio_features"):
            if self.audio_features_type == "mfcc":
                cleaned_mfcc = self.audio_features(cleaned_wav.cpu())
                original_mfcc = self.audio_features(original_wav.cpu())
                mfcc_diff = (cleaned_mfcc - original_mfcc).mean(2).detach().cpu().numpy()[0]
                if np.isnan(mfcc_diff).any():
                    mfcc_diff = np.zeros_like(mfcc_diff)
                return mfcc_diff
            elif self.audio_features_type == "hubert":
                cleaned_hubert = self.audio_features._get_embedding(cleaned_wav, 16000, text)
                original_hubert = self.audio_features._get_embedding(original_wav, 16000, text)
                hubert_diff = cleaned_hubert - original_hubert
                return hubert_diff
            elif self.audio_features_type == "wav2vec2":
                cleaned_wav2vec2 = self.audio_features._get_embedding(cleaned_wav, 16000, text)
                original_wav2vec2 = self.audio_features._get_embedding(original_wav, 16000, text)
                wav2vec2_diff = cleaned_wav2vec2 - original_wav2vec2
                return wav2vec2_diff


class Voicefixer(EmbeddingModel):
    def __init__(self, device="cuda", audio_features="mfcc", audio_features_device="cuda"):
        self.voicefixer = VoiceFixer()
        self.device = device
        self.audio_features_type = audio_features
        self.audio_features_device = audio_features_device
        if audio_features == "mfcc":
            self.audio_features = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
            )
        elif audio_features == "hubert":
            self.audio_features = Hubert(device=audio_features_device)
        elif audio_features == "wav2vec2":
            self.audio_features = Wav2Vec2(device=audio_features_device)

    def _get_embedding(self, audio, sr, text, speaker=None):
        # input & output temp files
        with tempfile.NamedTemporaryFile(suffix=".wav") as f_in:
            torchaudio.save(f_in.name, audio, sr)
            with tempfile.NamedTemporaryFile(suffix=".wav") as f_out:
                self.voicefixer.restore(f_in.name, f_out.name, cuda="cuda" in self.device, mode=0)
                # get difference between original and cleaned
                cleaned_wav, sr_new = torchaudio.load(f_out.name)
                cleaned_wav = cleaned_wav / cleaned_wav.abs().max()
                cleaned_wav = torchaudio.transforms.Resample(sr_new, 16000)(cleaned_wav)
                original_wav = audio
                original_wav = original_wav / original_wav.abs().max()
                if hasattr(self, "audio_features"):
                    if self.audio_features_type == "mfcc":
                        cleaned_mfcc = self.audio_features(cleaned_wav)
                        original_mfcc = self.audio_features(original_wav.cpu())
                        if cleaned_mfcc.shape[2] != original_mfcc.shape[2]:
                            cleaned_mfcc = cleaned_mfcc[:, :, :original_mfcc.shape[2]]
                        mfcc_diff = (cleaned_mfcc - original_mfcc).mean(2).detach().cpu().numpy()[0]
                        return mfcc_diff
                    elif self.audio_features_type == "hubert":
                        cleaned_hubert = self.audio_features._get_embedding(cleaned_wav, 16000, text)
                        original_hubert = self.audio_features._get_embedding(original_wav, 16000, text)
                        hubert_diff = cleaned_hubert - original_hubert
                        return hubert_diff
                    elif self.audio_features_type == "wav2vec2":
                        cleaned_wav2vec2 = self.audio_features._get_embedding(cleaned_wav, 16000, text)
                        original_wav2vec2 = self.audio_features._get_embedding(original_wav, 16000, text)
                        wav2vec2_diff = cleaned_wav2vec2 - original_wav2vec2
                        return wav2vec2_diff

class ProsodyMPM(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.mpm = MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")
        self.device = device
        self.mpm = self.mpm.to(device)
        self.mpm.eval()
        self.pitch_measure = PitchMeasure()
        self.energy_measure = EnergyMeasure()
        self.voice_activity_measure = VoiceActivityMeasure()
        self.pitch_min = 50
        self.pitch_max = 300
        self.energy_min = 0
        self.energy_max = 0.2
        self.vad_min = 0
        self.vad_max = 1
        self.bins = torch.linspace(0, 1, 128)

    def _get_embedding(self, audio, sr, text, speaker=None):
        # resample to 22050 if needed
        if sr != 22050:
            audio = torchaudio.transforms.Resample(sr, 22050)(audio)
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if len(audio.shape) == 2:
            audio = audio[0]
        pitch = self.pitch_measure(audio, np.array([1000]))["measure"]
        energy = self.energy_measure(audio, np.array([1000]))["measure"]
        vad = self.voice_activity_measure(audio, np.array([1000]))["measure"]
        pitch = torch.tensor(pitch)
        energy = torch.tensor(energy)
        vad = torch.tensor(vad)
        pitch[torch.isnan(pitch)] = -1000
        energy[torch.isnan(energy)] = -1000
        vad[torch.isnan(vad)] = -1000
        pitch = torch.clip(pitch, self.pitch_min, self.pitch_max)
        energy = torch.clip(energy, self.energy_min, self.energy_max)
        vad = torch.clip(vad, self.vad_min, self.vad_max)
        pitch = pitch / (self.pitch_max - self.pitch_min)
        energy = energy / (self.energy_max - self.energy_min)
        vad = vad / (self.vad_max - self.vad_min)
        pitch = torch.bucketize(pitch, self.bins)
        energy = torch.bucketize(energy, self.bins)
        vad = torch.bucketize(vad, torch.linspace(0, 1, 2))
        model_input = torch.stack([pitch, energy, vad]).unsqueeze(0)
        with torch.no_grad():
            reprs = self.mpm(model_input.to(self.device), return_layer=7)["representations"]
        reprs = reprs.mean(1).detach().cpu().numpy()[0]
        return reprs

wasserstein = lambda x, y: np.mean(np.abs(np.sort(x) - np.sort(y)))

class WERModel(ABC):
    def get_wer(self, audio, text, noise=False):
        if isinstance(audio, str) or isinstance(audio, Path):
            audio, sr = torchaudio.load(audio)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            sr = 16000
        if isinstance(audio, tuple):
            audio, sr = audio
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float().unsqueeze(0)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            sr = 16000
        if noise:
            audio = torch.randn_like(audio)
            audio = audio / audio.abs().max()
        # normalize text
        text = text.lower()
        transcript = self._get_transcript(audio, sr)
        transcript = transcript.lower()
        # remove punctuation
        transcript = re.sub(r'[^\w\s]', '', transcript)
        text = re.sub(r'[^\w\s]', '', text)
        # remove multiple spaces
        transcript = re.sub(r'\s+', ' ', transcript)
        text = re.sub(r'\s+', ' ', text)
        wer_result = wer(text, transcript)
        return wer_result


    def get_wer_dist(self, ds_name):
        prefix = self.__class__.__name__.lower()
        if (Path("cache") / f"{prefix}_{ds_name}_wer.npy").exists():
            return np.load(Path("cache") / f"{prefix}_{ds_name}_wer.npy")
        wer = []
        audios, texts = self.get_audios(ds_name)
        for audio, text in tqdm(zip(audios, texts), f"generating {ds_name}"):
            wer.append(self.get_wer(audio, text))
        np.save(Path("cache") / f"{prefix}_{ds_name}_wer.npy", wer)
        return wer

    def get_noise_wer_dist(self):
        prefix = self.__class__.__name__.lower()
        if (Path("cache") / f"{prefix}_noise_wer.npy").exists():
            return np.load(Path("cache") / f"{prefix}_noise_wer.npy")
        wer = []
        audios, texts = self.get_audios("reference.test")
        for audio, text in tqdm(zip(audios, texts), f"generating noise"):
            wer.append(self.get_wer(audio, text, noise=True))
        np.save(Path("cache") / f"{prefix}_noise_wer.npy", wer)
        return wer

    def get_audios(self, ds_name):
        return generate_data(ds_name, self.device)

    def get_wasserstein(self, ds_name):
        result = np.array(
            [
                wasserstein(
                    self.get_wer_dist(ds_name),
                    self.get_wer_dist("reference.test")
                ),
                wasserstein(
                    self.get_wer_dist(ds_name),
                    self.get_wer_dist("reference.dev")
                ),
            ]
        )
        worst_value = wasserstein(
            self.get_wer_dist("reference.test"),
            self.get_noise_wer_dist()
        )
        best_value = wasserstein(
            self.get_wer_dist("reference.test"),
            self.get_wer_dist("reference.dev"),
        )
        # convert to score from 0 to 1
        result = 1 - ((result - best_value) / (worst_value - best_value))
        return np.round(np.min(result * 100), 2)

    @abstractmethod
    def _get_transcript(self, audio, sr):
        raise NotImplementedError

class Whisper(WERModel):
    def __init__(self, device="cuda"):
        self.whisper = whisper.load_model("small.en")
        self.whisper = self.whisper.to(device)
        self.device = device

    def _get_transcript(self, audio, sr):
        # resample to 16k
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, audio, sr)
            transcript = self.whisper.transcribe(f.name)
            return transcript["text"]

class Wav2Vec2WER(WERModel):
    def __init__(self, device="cuda"):
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = wav2vec2_model.to(device)
        self.wav2vec2_model.eval()
        self.device = device

    def _get_transcript(self, audio, sr):
        input_values = self.wav2vec2_processor(
            audio[0], return_tensors="pt", sampling_rate=16000
        ).input_values
        with torch.no_grad():
            logits = self.wav2vec2_model(input_values.to(self.device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec2_processor.batch_decode(predicted_ids)
        return transcription[0]