from abc import ABC, abstractmethod
import os
import tempfile

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pathlib import Path
from tqdm.auto import tqdm
from pyannote.audio import Model
from dotenv import load_dotenv
from pyannote.audio import Inference
from datasets import load_dataset
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from miipher.lightning_module import MiipherLightningModule

from generate_data import generate_parler_tts, generate_hifigan, generate_xtts
from frechet_distance import frechet_distance
from dvector.wav2mel import Wav2Mel

load_dotenv()

class EmbeddingModel(ABC):
    model_layer = 6

    def get_embedding(self, audio, noise=False):
        if isinstance(audio, str) or isinstance(audio, Path):
            audio, sr = torchaudio.load(audio)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        if isinstance(audio, tuple):
            audio, sr = audio
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float().unsqueeze(0)
            audio = audio / audio.abs().max()
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        if noise:
            audio = torch.randn_like(audio)
            audio = audio / audio.abs().max()
        return self._get_embedding(audio, sr)

    def get_mu_sigma(self, ds_name):
        noise = False
        prefix = self.__class__.__name__.lower()
        if (Path("cache") / f"{prefix}_{ds_name}_mu.npy").exists():
            mu = np.load(Path("cache") / f"{prefix}_{ds_name}_mu.npy")
            sigma = np.load(Path("cache") / f"{prefix}_{ds_name}_sigma.npy")
            return mu, sigma
        features = []
        audios = self.get_audios(ds_name)
        for audio in tqdm(audios, f"generating {ds_name}"):
            features.append(self.get_embedding(audio, noise=noise))
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
        audios = self.get_audios("reference.test")
        for audio in tqdm(audios, f"generating {ds_name}"):
            features.append(self.get_embedding(audio, noise=noise))
        features = np.array(features)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        np.save(Path("cache") / f"{prefix}_{ds_name}_mu.npy", mu)
        np.save(Path("cache") / f"{prefix}_{ds_name}_sigma.npy", sigma)
        return mu, sigma

    def get_audios(self, ds_name):
        if ds_name == "reference.test":
            ds = load_dataset(
                "cdminix/libritts-aligned", split="test.clean", trust_remote_code=True
            )
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(1000))
            return [item["audio"] for item in ds]
        elif ds_name == "reference.dev":
            ds = load_dataset(
                "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
            )
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(1000))
            return [item["audio"] for item in ds]
        elif ds_name == "parler":
            return generate_parler_tts(self.device)
        elif ds_name == "hifigan":
            return generate_hifigan()
        elif ds_name == "xtts":
            return generate_xtts(self.device)

    def get_frechet(self, ds_name):
        result = np.array(
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
        result = 1 - ((result - best_value) / (worst_value - best_value))
        return np.min(result * 100)
    

    @abstractmethod
    def _get_embedding(self, audio, sr):
        pass


class Hubert(EmbeddingModel):
    def __init__(self, device="cuda"):
        self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_model = hubert_model.to(device)
        self.hubert_model.eval()
        self.device = device

    def _get_embedding(self, audio, sr):
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

    def _get_embedding(self, audio, sr):
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

    def _get_embedding(self, audio, sr):
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

    def _get_embedding(self, audio, sr):
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
    def __init__(self, device="cuda"):
        miipher_path = "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/miipher.ckpt"
        self.miipher = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location=device)
        self.preprocessor = PreprocessForInfer(self.miipher.cfg)

    def _get_embedding(self, audio, sr, text):
        wav = audio
        batch = self.preprocessor.process(
            'test',
            (torch.tensor(wav),sr),
            word_segmented_text=transcript,
            lang_code="eng-us"
        )
        self.miipher.feature_extractor(batch)
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            _,
        ) = self.miipher.feature_extractor(batch)
        cleaned_ssl_feature, _ = self.miipher(phone_feature, speaker_feature, degraded_ssl_feature)
        clean_deg_diff = torch.abs(cleaned_ssl_feature - degraded_ssl_feature)
        return clean_deg_diff


if __name__ == "__main__":
    hubert = Hubert()
    print("Hubert")
    print("Reference test:", hubert.get_frechet("reference.test"))
    print("Reference dev:", hubert.get_frechet("reference.dev"))
    print("Parler:", hubert.get_frechet("parler"))
    print("Hifigan:", hubert.get_frechet("hifigan"))
    print("Xtts:", hubert.get_frechet("xtts"))

    wav2vec2 = Wav2Vec2()
    print("Wav2Vec2")
    print("Reference test:", wav2vec2.get_frechet("reference.test"))
    print("Reference dev:", wav2vec2.get_frechet("reference.dev"))
    print("Parler:", wav2vec2.get_frechet("parler"))
    print("Hifigan:", wav2vec2.get_frechet("hifigan"))
    print("Xtts:", wav2vec2.get_frechet("xtts"))

    dvector = DVector()
    print("DVector")
    print("Reference test:", dvector.get_frechet("reference.test"))
    print("Reference dev:", dvector.get_frechet("reference.dev"))
    print("Parler:", dvector.get_frechet("parler"))
    print("Hifigan:", dvector.get_frechet("hifigan"))
    print("Xtts:", dvector.get_frechet("xtts"))

    xvector = XVector()
    print("XVector")
    print("Reference test:", xvector.get_frechet("reference.test"))
    print("Reference dev:", xvector.get_frechet("reference.dev"))
    print("Parler:", xvector.get_frechet("parler"))
    print("Hifigan:", xvector.get_frechet("hifigan"))
    print("Xtts:", xvector.get_frechet("xtts"))

    miipher = Miipher()
    print("Miipher")
    print("Reference test:", miipher.get_frechet("reference.test"))
    print("Reference dev:", miipher.get_frechet("reference.dev"))
    print("Parler:", miipher.get_frechet("parler"))
    print("Hifigan:", miipher.get_frechet("hifigan"))
    print("Xtts:", miipher.get_frechet("xtts"))