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

from generate_data import generate_parler_tts, generate_hifigan, generate_xtts, N_SAMPLES
from frechet_distance import frechet_distance
from dvector.wav2mel import Wav2Mel

load_dotenv()


class EmbeddingModel(ABC):
    model_layer = 6

    def get_embedding(self, audio, text, noise=False):
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
        return self._get_embedding(audio, sr, text)

    def get_mu_sigma(self, ds_name):
        noise = False
        prefix = self.__class__.__name__.lower()
        if (Path("cache") / f"{prefix}_{ds_name}_mu.npy").exists():
            mu = np.load(Path("cache") / f"{prefix}_{ds_name}_mu.npy")
            sigma = np.load(Path("cache") / f"{prefix}_{ds_name}_sigma.npy")
            return mu, sigma
        features = []
        audios, texts = self.get_audios(ds_name)
        for audio, text in tqdm(zip(audios, texts), f"generating {ds_name}"):
            features.append(self.get_embedding(audio, text, noise=noise))
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        np.save(Path("cache") / f"{prefix}_{ds_name}_mu.npy", mu)
        np.save(Path("cache") / f"{prefix}_{ds_name}_sigma.npy", sigma)
        return mu, sigma

    def get_noise_mu_sigma(self):
        noise = True
        ds_name = "noise"
        prefix = self.__class__.__name__.lower()
        if (Path("cache") / f"{prefix}_{ds_name}_mu.npy").exists():
            mu = np.load(Path("cache") / f"{prefix}_{ds_name}_mu.npy")
            sigma = np.load(Path("cache") / f"{prefix}_{ds_name}_sigma.npy")
            return mu, sigma
        features = []
        audios, texts = self.get_audios("reference.test")
        for audio, text in tqdm(zip(audios, texts), f"generating {ds_name}"):
            features.append(self.get_embedding(audio, text, noise=noise))
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        np.save(Path("cache") / f"{prefix}_{ds_name}_mu.npy", mu)
        np.save(Path("cache") / f"{prefix}_{ds_name}_sigma.npy", sigma)
        return mu, sigma

    def get_audios(self, ds_name):
        if ds_name == "reference.test":
            return generate_ref_test()
        elif ds_name == "reference.dev":
            return generate_ref_dev()
        elif ds_name == "parler":
            return generate_parler_tts(self.device)
        elif ds_name == "hifigan":
            return generate_hifigan()
        elif ds_name == "xtts":
            return generate_xtts(self.device)

    def get_frechet(self, ds_name):
        val = np.array(
            [
                frechet_distance(
                    *self.get_mu_sigma(ds_name),
                    *self.get_mu_sigma(f"reference.test")
                ),
                frechet_distance(
                    *self.get_mu_sigma(ds_name),
                    *self.get_mu_sigma(f"reference.dev")
                ),
            ]
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
        return np.round(np.min(result * 100), 2), np.round(np.max(val), 3)
    

    @abstractmethod
    def _get_embedding(self, audio, sr, text):
        pass


class MFCC(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.device = device
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
        )

    def _get_embedding(self, audio, sr, text):
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

    def _get_embedding(self, audio, sr, text):
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

    def _get_embedding(self, audio, sr, text):
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

    def _get_embedding(self, audio, sr, text):
        mel_tensor = self.wav2mel(audio, sr)  # shape: (frames, mel_dim)
        mel_tensor = mel_tensor.to(self.device)
        emb_tensor = self.dvector.embed_utterance(mel_tensor)
        return emb_tensor.detach().cpu().numpy()[0]

class XVector(EmbeddingModel):
    def __init__(self, device="cuda"):
        xvector_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        xvector_model.to(device)
        self.xvector = Inference(xvector_model, window="whole")
        self.device = device

    def _get_embedding(self, audio, sr, text):
        # save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, audio, sr)
            # get embedding
            try:
                embedding = self.xvector(f.name)
            except RuntimeError:
                embedding = np.zeros(512)
        return embedding[0]
    
class Miipher(EmbeddingModel):
    """
    See: https://github.com/Wataru-Nakata/miipher/blob/main/examples/demo.py
    """

    def __init__(self, device="cuda", audio_features="mfcc"):
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
        if audio_features == "mfcc":
            self.audio_features = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
            )

    def _get_embedding(self, audio, sr, text):
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
            cleaned_mfcc = self.audio_features(cleaned_wav.cpu())
            original_mfcc = self.audio_features(original_wav.cpu())
            mfcc_diff = (cleaned_mfcc - original_mfcc).mean(2).detach().cpu().numpy()[0]
            return mfcc_diff

class Voicefixer(EmbeddingModel):
    def __init__(self, device="cuda", audio_features="mfcc"):
        self.voicefixer = VoiceFixer()
        self.device = device
        if audio_features == "mfcc":
            self.audio_features = torchaudio.transforms.MFCC(
                sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
            )

    def _get_embedding(self, audio, sr, text):
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
                    cleaned_mfcc = self.audio_features(cleaned_wav)
                    original_mfcc = self.audio_features(original_wav.cpu())
                    mfcc_diff = (cleaned_mfcc - original_mfcc).mean(2).detach().cpu().numpy()[0]
                    return mfcc_diff

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
        if ds_name == "reference.test":
            ds = load_dataset(
                "cdminix/libritts-aligned", split="test.clean", trust_remote_code=True
            )
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(N_SAMPLES))
            return [item["audio"] for item in ds], [item["text"] for item in ds]
        elif ds_name == "reference.dev":
            ds = load_dataset(
                "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
            )
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(N_SAMPLES))
            return [item["audio"] for item in ds], [item["text"] for item in ds]
        elif ds_name == "parler":
            return generate_parler_tts(self.device)
        elif ds_name == "hifigan":
            return generate_hifigan()
        elif ds_name == "xtts":
            return generate_xtts(self.device)

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

if __name__ == "__main__":
    # mfcc = MFCC()
    # print("MFCC")
    # print("Reference test:", mfcc.get_frechet("reference.test"))
    # print("Reference dev:", mfcc.get_frechet("reference.dev"))
    # print("Parler:", mfcc.get_frechet("parler"))
    # print("Hifigan:", mfcc.get_frechet("hifigan"))
    # print("Xtts:", mfcc.get_frechet("xtts"))

    # hubert = Hubert()
    # print("Hubert")
    # print("Reference test:", hubert.get_frechet("reference.test"))
    # print("Reference dev:", hubert.get_frechet("reference.dev"))
    # print("Parler:", hubert.get_frechet("parler"))
    # print("Hifigan:", hubert.get_frechet("hifigan"))
    # print("Xtts:", hubert.get_frechet("xtts"))

    # wav2vec2 = Wav2Vec2()
    # print("Wav2Vec2")
    # print("Reference test:", wav2vec2.get_frechet("reference.test"))
    # print("Reference dev:", wav2vec2.get_frechet("reference.dev"))
    # print("Parler:", wav2vec2.get_frechet("parler"))
    # print("Hifigan:", wav2vec2.get_frechet("hifigan"))
    # print("Xtts:", wav2vec2.get_frechet("xtts"))

    # dvector = DVector()
    # print("DVector")
    # print("Reference test:", dvector.get_frechet("reference.test"))
    # print("Reference dev:", dvector.get_frechet("reference.dev"))
    # print("Parler:", dvector.get_frechet("parler"))
    # print("Hifigan:", dvector.get_frechet("hifigan"))
    # print("Xtts:", dvector.get_frechet("xtts"))

    # xvector = XVector()
    # print("XVector")
    # print("Reference test:", xvector.get_frechet("reference.test"))
    # print("Reference dev:", xvector.get_frechet("reference.dev"))
    # print("Parler:", xvector.get_frechet("parler"))
    # print("Hifigan:", xvector.get_frechet("hifigan"))
    # print("Xtts:", xvector.get_frechet("xtts"))

    # miipher = Miipher()
    # print("Miipher")
    # print("Reference test:", miipher.get_frechet("reference.test"))
    # print("Reference dev:", miipher.get_frechet("reference.dev"))
    # print("Parler:", miipher.get_frechet("parler"))
    # print("Hifigan:", miipher.get_frechet("hifigan"))
    # print("Xtts:", miipher.get_frechet("xtts"))

    voicefixer = Voicefixer("cpu")
    print("Voicefixer")
    print("Reference test:", voicefixer.get_frechet("reference.test"))
    print("Reference dev:", voicefixer.get_frechet("reference.dev"))
    print("Parler:", voicefixer.get_frechet("parler"))
    print("Hifigan:", voicefixer.get_frechet("hifigan"))
    print("Xtts:", voicefixer.get_frechet("xtts"))

    # whisper = Whisper()
    # print("Whisper")
    # print("Reference test:", whisper.get_wasserstein("reference.test"))
    # print("Reference dev:", whisper.get_wasserstein("reference.dev"))
    # print("Parler:", whisper.get_wasserstein("parler"))
    # print("Hifigan:", whisper.get_wasserstein("hifigan"))
    # print("Xtts:", whisper.get_wasserstein("xtts"))

    # wav2vec2_wer = Wav2Vec2WER()
    # print("Wav2Vec2WER")
    # print("Reference test:", wav2vec2_wer.get_wasserstein("reference.test"))
    # print("Reference dev:", wav2vec2_wer.get_wasserstein("reference.dev"))
    # print("Parler:", wav2vec2_wer.get_wasserstein("parler"))
    # print("Hifigan:", wav2vec2_wer.get_wasserstein("hifigan"))
    # print("Xtts:", wav2vec2_wer.get_wasserstein("xtts"))