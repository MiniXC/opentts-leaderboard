import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertModel
from generate_data import generate_data
from tqdm import tqdm
from sklearn.cluster import KMeans
from pathlib import Path

class HubertUnitCounter:
    def __init__(self):
        self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_model.eval()
        self.model_layer = 7
        self.km = self.get_reference_clusters()

    def get_reference_clusters(self):
        if Path("data/reference_clusters.pt").exists():
            clusters = torch.load("data/reference_clusters.pt")
            km = KMeans(n_clusters=100, init=clusters)
            km.fit(clusters) # this is a hack
            km.cluster_centers_ = clusters
            return km
        audios, texts = generate_data("reference.test")
        km = KMeans(n_clusters=100)
        clusters = []
        all_features = []
        for audio in tqdm(audios, desc="Generating reference clusters"):
            audio, sr = torchaudio.load(audio)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            input_values = self.hubert_processor(
                audio, return_tensors="pt", sampling_rate=16000
            ).input_values
            with torch.no_grad():
                features = self.hubert_model(input_values[0], output_hidden_states=True).hidden_states
            features = features[self.model_layer].cpu().numpy()[0]
            all_features.append(features)
        all_features = np.concatenate(all_features)
        km.fit(all_features)
        clusters = km.cluster_centers_
        torch.save(clusters, "data/reference_clusters.pt")
        return km

    def count_units(self, ds):
        if Path(f"data/{ds}_units.pt").exists():
            return torch.load(f"data/{ds}_unit_counts.pt"), torch.load(f"data/{ds}_unit_lengths.pt")
        audios, _ = generate_data(ds)
        unit_counts = {}
        unit_lengths = {}
        for audio in tqdm(audios, desc=f"Counting units for {ds}"):
            audio, sr = torchaudio.load(audio)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            input_values = self.hubert_processor(
                audio, return_tensors="pt", sampling_rate=16000
            ).input_values
            with torch.no_grad():
                features = self.hubert_model(input_values[0], output_hidden_states=True).hidden_states
            features = features[self.model_layer].cpu().numpy()[0]
            predictions = self.km.predict(features)
            for unit in predictions:
                if unit not in unit_counts:
                    unit_counts[unit] = 0
                unit_counts[unit] += 1
            current_unit = predictions[0]
            current_length = 1
            for unit in predictions[1:]:
                if unit != current_unit:
                    if current_unit not in unit_lengths:
                        unit_lengths[current_unit] = []
                    unit_lengths[current_unit].append(current_length)
                    current_length = 1
                    current_unit = unit
                else:
                    current_length += 1
            if current_unit not in unit_lengths:
                unit_lengths[current_unit] = []
            unit_lengths[current_unit].append(current_length)
        torch.save(unit_counts, f"data/{ds}_units.pt")
        torch.save(unit_lengths, f"data/{ds}_unit_lengths.pt")
        return unit_counts, unit_lengths


import matplotlib.pyplot as plt

hubert = HubertUnitCounter()
reference_units, reference_l = hubert.count_units("reference.test")
parler_units, parler_l = hubert.count_units("parler")

plt.bar(range(100), [reference_units[i] for i in range(100)], color="blue", alpha=0.5, label="Reference")
plt.bar(range(100), [parler_units[i] for i in range(100)], color="red", alpha=0.5, label="Parler")
plt.legend()
plt.savefig("units.png")
plt.close()

all_reference_lengths = [
    item for sublist in reference_l.values() for item in sublist
]
all_parler_lengths = [
    item for sublist in parler_l.values() for item in sublist
]
# histogram of unit lengths
plt.hist(all_reference_lengths, bins=100, color="blue", alpha=0.5, label="Reference")
plt.hist(all_parler_lengths, bins=100, color="red", alpha=0.5, label="Parler")
plt.legend()
plt.savefig("unit_lengths.png")
plt.close()
