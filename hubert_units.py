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
        if Path(f"data/{ds}_unit_counts.pt").exists():
            unit_counts, unit_lengths = torch.load(f"data/{ds}_unit_counts.pt"), torch.load(f"data/{ds}_unit_lengths.pt")
            unit_counts = {
                k: v / sum(unit_counts.values()) * 100 for k, v in unit_counts.items()
            }
            return unit_counts, unit_lengths
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
        torch.save(unit_counts, f"data/{ds}_unit_counts.pt")
        torch.save(unit_lengths, f"data/{ds}_unit_lengths.pt")
        unit_counts = {
            k: v / sum(unit_counts.values()) * 100 for k, v in unit_counts.items()
        }
        return unit_counts, unit_lengths

    def get_wasserstein_distances(self, ds):
        test_c, test_l = hubert.count_units("reference.test")
        dev_c, dev_l = hubert.count_units("reference.dev")
        ds_c, ds_l = hubert.count_units(ds)
        all_test_lengths = [
            item for sublist in test_l.values() for item in sublist
        ]
        all_dev_lengths = [
            item for sublist in dev_l.values() for item in sublist
        ]
        all_ds_lengths = [
            item for sublist in ds_l.values() for item in sublist
        ]
        all_test_lengths = np.array(all_test_lengths)
        all_dev_lengths = np.array(all_dev_lengths)
        all_ds_lengths = np.array(all_ds_lengths)
        # take a random sample of whichever is the smallest
        min_length = min(len(all_test_lengths), len(all_dev_lengths), len(all_ds_lengths))
        np.random.seed(42)
        all_test_lengths = np.random.choice(all_test_lengths, min_length)
        np.random.seed(42)
        all_dev_lengths = np.random.choice(all_dev_lengths, min_length)
        np.random.seed(42)
        all_ds_lengths = np.random.choice(all_ds_lengths, min_length)
        all_test_lengths = np.sort(all_test_lengths)
        all_dev_lengths = np.sort(all_dev_lengths)
        all_ds_lengths = np.sort(all_ds_lengths)
        mean_length_repeated = np.ones_like(all_test_lengths) * np.mean(all_test_lengths)
        best_wasserstein = np.abs(all_test_lengths - all_dev_lengths).mean()
        worst_wassterstein = np.abs(all_test_lengths - mean_length_repeated).mean()
        length_wasserstein = np.abs(all_ds_lengths - all_test_lengths).mean()
        # scale so that 100 is best_wasserstein and 0 is worst_wasserstein
        length_wasserstein = (1 - (length_wasserstein - best_wasserstein) / (worst_wassterstein - best_wasserstein)) * 100
        all_test_counts = np.array([
            test_c[i] for i in range(100)
        ])
        all_dev_counts = np.array([
            dev_c[i] for i in range(100)
        ])
        all_ds_counts = np.array([
            ds_c[i] for i in range(100)
        ])
        all_test_counts = np.sort(all_test_counts)
        all_dev_counts = np.sort(all_dev_counts)
        all_ds_counts = np.sort(all_ds_counts)
        mean_unit_values = np.ones_like(all_test_counts) * np.mean(all_test_counts)
        best_wasserstein = np.abs(all_test_counts - all_dev_counts).mean()
        worst_wassterstein = np.abs(all_test_counts - mean_unit_values).mean()
        unit_wasserstein = np.abs(all_ds_counts - all_test_counts).mean()
        # scale so that 100 is best_wasserstein and 0 is worst_wasserstein
        unit_wasserstein = (1 - (unit_wasserstein - best_wasserstein) / (worst_wassterstein - best_wasserstein)) * 100
        return length_wasserstein, unit_wasserstein

hubert = HubertUnitCounter()
print(hubert.get_wasserstein_distances("reference.dev"))
print(hubert.get_wasserstein_distances("parler"))