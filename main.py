from datasets import load_dataset
from transformers import Wav2Vec2Processor, HubertModel
import torchaudio
import numpy as np
from scipy import linalg
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from pathlib import Path


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    From: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


reference_test = load_dataset(
    "cdminix/libritts-aligned", split="test.clean", trust_remote_code=True
)
reference_val = load_dataset(
    "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
)

# shuffling the dataset
reference_test = reference_test.shuffle()
reference_val = reference_val.shuffle()

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

print("loading model")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
print("model loaded")


def get_hubert_embedding(audio, layer=6):
    audio, sr = torchaudio.load(audio)
    audio = audio / audio.abs().max()
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    input_values = processor(
        audio, return_tensors="pt", sampling_rate=16000
    ).input_values
    features = model(input_values[0], output_hidden_states=True).hidden_states
    features = features[layer].mean(1).detach().numpy()[0]
    return features


def get_hubert_mu_sigma(audios, layer=6, cache_name=None):
    if cache_name and (Path("cache") / "{cache_name}_mu.npy").exists():
        mu = np.load(Path("cache") / f"{cache_name}_mu.npy")
        sigma = np.load(Path("cache") / f"{cache_name}_sigma.npy")
        return mu, sigma
    features = []
    for audio in audios:
        features.append(get_hubert_embedding(audio, layer=layer))
    features = np.array(features)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.save(Path("cache") / f"{cache_name}_{layer}_mu.npy", mu)
    np.save(Path("cache") / f"{cache_name}_{layer}_sigma.npy", sigma)
    return mu, sigma


test_mu, test_sigma = get_hubert_mu_sigma(
    [item["audio"] for item in tqdm(reference_test)], cache_name="test"
)
val_mu, val_sigma = get_hubert_mu_sigma(
    [item["audio"] for item in tqdm(reference_val)], cache_name="val"
)

print(f"frechet distance: {frechet_distance(test_mu, test_sigma, val_mu, val_sigma)}")
