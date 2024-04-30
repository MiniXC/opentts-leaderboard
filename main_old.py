from datasets import load_dataset
from transformers import Wav2Vec2Processor, HubertModel, Wav2Vec2Model
import torchaudio
import torch
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from pathlib import Path
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import requests
from simple_hifigan import Synthesiser
import os

from frechet_distance import frechet_distance

os.environ["LIBRITTS_DOWNLOAD_SPLITS"] = "dev-clean,dev-other,test-clean,test-other"

reference_test = load_dataset(
    "cdminix/libritts-aligned", split="test.clean", trust_remote_code=True
)
reference_val = load_dataset(
    "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
)
parler_val = load_dataset("parler-tts/libritts_r_tags_tagged_10k_generated", "clean", split="test.clean")
librittsr_val = load_dataset("blabble-io/libritts_r", "dev", split="dev.clean")

# shuffling the dataset
reference_test = reference_test.shuffle(seed=42)
reference_val = reference_val.shuffle(seed=42)
parler_val = parler_val.shuffle(seed=42)
librittsr_val = librittsr_val.shuffle(seed=42)

# randomly sample the dataset (1000 samples)
reference_test = reference_test.select(range(1000))
reference_val = reference_val.select(range(1000))
parler_val = parler_val.select(range(1000))
librittsr_val = librittsr_val.select(range(1000))

hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert_model = hubert_model.to("cuda")
hifigan = Synthesiser()

wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = wav2vec2_model.to("cuda")

parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to("cuda")
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

Path("cache").mkdir(exist_ok=True, parents=True)
Path("data").mkdir(exist_ok=True, parents=True)

def resynthesize(audio_path):
    data_path = Path("data/hifigan") / Path(audio_path).name
    if data_path.exists():
        return data_path
    data_path.parent.mkdir(exist_ok=True, parents=True)
    audio, sr = torchaudio.load(audio_path)
    mel_spectrogram = hifigan.wav_to_mel(
        audio, sr
    )
    audio = torch.tensor(hifigan(mel_spectrogram.squeeze(0)))
    # to float
    audio = audio.float() / audio.float().abs().max()
    torchaudio.save(data_path, audio, 22050)
    return data_path

def get_hubert_embedding(audio, layer=6, noise=False):
    if isinstance(audio, str):
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
    input_values = hubert_processor(
        audio, return_tensors="pt", sampling_rate=16000
    ).input_values
    with torch.no_grad():
      features = hubert_model(input_values[0].to("cuda"), output_hidden_states=True).hidden_states
    features = features[layer].mean(1).detach().cpu().numpy()[0]
    return features


def get_hubert_mu_sigma(audios, layer=6, cache_name=None, noise=False):
    prefix = "hubert"
    if cache_name and (Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy").exists():
        mu = np.load(Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy")
        sigma = np.load(Path("cache") / f"{prefix}_{cache_name}_{layer}_sigma.npy")
        return mu, sigma
    features = []
    for audio in tqdm(audios, f"generating {cache_name}"):
        features.append(get_hubert_embedding(audio, layer=layer, noise=noise))
    features = np.array(features)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.save(Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy", mu)
    np.save(Path("cache") / f"{prefix}_{cache_name}_{layer}_sigma.npy", sigma)
    return mu, sigma


def get_wav2vec2_embedding(audio, layer=6, noise=False):
    if isinstance(audio, str):
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
    input_values = wav2vec2_processor(
        audio, return_tensors="pt", sampling_rate=16000
    ).input_values
    with torch.no_grad():
      features = wav2vec2_model(input_values[0].to("cuda"), output_hidden_states=True).hidden_states
    features = features[layer].mean(1).detach().cpu().numpy()[0]
    return features


def get_wav2vec2_mu_sigma(audios, layer=6, cache_name=None, noise=False):
    prefix = "w2v2"
    if cache_name and (Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy").exists():
        mu = np.load(Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy")
        sigma = np.load(Path("cache") / f"{prefix}_{cache_name}_{layer}_sigma.npy")
        return mu, sigma
    features = []
    for audio in tqdm(audios, f"generating {cache_name}"):
        features.append(get_wav2vec2_embedding(audio, layer=layer, noise=noise))
    features = np.array(features)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.save(Path("cache") / f"{prefix}_{cache_name}_{layer}_mu.npy", mu)
    np.save(Path("cache") / f"{prefix}_{cache_name}_{layer}_sigma.npy", sigma)
    return mu, sigma


test_mu, test_sigma = get_hubert_mu_sigma(
    [item["audio"] for item in tqdm(reference_test)], layer=6, cache_name="test"
)
val_mu, val_sigma = get_hubert_mu_sigma(
    [item["audio"] for item in tqdm(reference_val)], layer=6, cache_name="val"
)
hubert_frechet_min = frechet_distance(test_mu, test_sigma, val_mu, val_sigma)
print(f"min. expected hubert frechet distance: {hubert_frechet_min}")

noise_mu, noise_sigma = get_hubert_mu_sigma(
    [item["audio"] for item in tqdm(reference_test)], layer=6, noise=True, cache_name="noise"
)
hubert_frechet_max = frechet_distance(test_mu, test_sigma, noise_mu, noise_sigma)
print(f"max. expected hubert frechet distance: {hubert_frechet_max}")

parler_real_mu, parler_real_sigma = get_hubert_mu_sigma(
    [
        (item["audio"]["array"], item["audio"]["sampling_rate"]) 
        for item in tqdm(librittsr_val)
    ], 
    layer=6, 
    cache_name="parler_real"
)
parler_real_frechet = frechet_distance(test_mu, test_sigma, parler_real_mu, parler_real_sigma)
# use min and max to convert to a score from 0 to 1
parler_real_score = 1 - ((parler_real_frechet - hubert_frechet_min) / (hubert_frechet_max - hubert_frechet_min))
print(f"hubert frechet distance (parler real): {parler_real_score * 100:.2f}")

# hifi-gan
hifigan_audios = [resynthesize(item["audio"]) for item in tqdm(reference_val, desc="resynthesizing")]
hifigan_mu, hifigan_sigma = get_hubert_mu_sigma(
    [str(x) for x in hifigan_audios], layer=6, cache_name="hifigan"
)
hifigan_frechet = frechet_distance(test_mu, test_sigma, hifigan_mu, hifigan_sigma)
# use min and max to convert to a score from 0 to 1
hifigan_score = 1 - ((hifigan_frechet - hubert_frechet_min) / (hubert_frechet_max - hubert_frechet_min))
print(hifigan_score, hifigan_frechet)
print(f"hubert frechet distance (hifigan): {hifigan_score * 100:.2f}")

# half precision
parler_model = parler_model.half()

Path("data/parler").mkdir(exist_ok=True, parents=True)

speaker_dict = {}

for item in tqdm(parler_val):
    if Path(f"data/parler/{item['id']}.wav").exists():
        continue
    prompt_input_ids = parler_tokenizer(item["text"], return_tensors="pt").input_ids.to("cuda")
    # we only use one description per speaker
    if item["speaker_id"] not in speaker_dict:
        input_ids = parler_tokenizer(item["text_description"], return_tensors="pt").input_ids
        speaker_dict[item["speaker_id"]] = input_ids
    input_ids = speaker_dict[item["speaker_id"]].to("cuda")
    with torch.no_grad():
        generated = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        if parler_model.config.sampling_rate != 16000:
            generated = torchaudio.transforms.Resample(parler_model.config.sampling_rate, 16000)(generated.to("cpu").float())
    generated = generated / generated.abs().max()
    torchaudio.save(f"data/parler/{item['id']}.wav", generated, 16000)

parler_audios = [f"data/parler/{item['id']}.wav" for item in parler_val]

parler_mu, parler_sigma = get_hubert_mu_sigma(
    parler_audios, layer=6, cache_name="parler"
)
parler_frechet = frechet_distance(val_mu, val_sigma, parler_mu, parler_sigma)
# use min and max to convert to a score from 0 to 1
parler_score = 1 - ((parler_frechet - hubert_frechet_min) / (hubert_frechet_max - hubert_frechet_min))
print(parler_score, parler_frechet)
print(f"hubert frechet distance (parler): {parler_score * 100:.2f}")