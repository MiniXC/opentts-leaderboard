from pathlib import Path

from tqdm.auto import tqdm
import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from simple_hifigan import Synthesiser
from TTS.api import TTS
import numpy as np

def generate_parler_tts(device="cuda"):
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    parler_val = load_dataset("parler-tts/libritts_r_tags_tagged_10k_generated", "clean", split="test.clean")
    parler_val = parler_val.shuffle(seed=42)
    parler_val = parler_val.select(range(1000))
    Path("data/parler").mkdir(exist_ok=True, parents=True)
    speaker_dict = {}
    for item in tqdm(parler_val, "generating parler tts"):
        if Path(f"data/parler/{item['id']}.wav").exists():
            continue
        prompt_input_ids = parler_tokenizer(item["text"], return_tensors="pt").input_ids.to(device)
        # we only use one description per speaker
        if item["speaker"] not in speaker_dict:
            input_ids = parler_tokenizer(item["text_description"], return_tensors="pt").input_ids
            speaker_dict[item["speaker"]] = input_ids
        input_ids = speaker_dict[item["speaker"]].to(device)
        with torch.no_grad():
            generated = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            if parler_model.config.sampling_rate != 16000:
                generated = torchaudio.transforms.Resample(parler_model.config.sampling_rate, 16000)(generated.to("cpu").float())
        generated = generated / generated.abs().max()
        torchaudio.save(f"data/parler/{item['id']}.wav", generated, 16000)
        with open(f"data/parler/{item['id']}.txt", "w") as f:
            f.write(item["text"])
    return [f"data/parler/{item['id']}.wav" for item in parler_val]

def generate_hifigan():
    hifigan = Synthesiser()
    results = []
    val = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    val = val.shuffle(seed=42)
    val = val.select(range(1000))
    for audio in tqdm(val, "generating hifigan"):
        audio_path = audio["audio"]
        data_path = Path("data/hifigan") / Path(audio_path).name
        if not data_path.exists():
            data_path.parent.mkdir(exist_ok=True, parents=True)
            audio, sr = torchaudio.load(audio_path)
            mel_spectrogram = hifigan.wav_to_mel(
                audio, sr
            )
            audio = torch.tensor(hifigan(mel_spectrogram.squeeze(0)))
            # to float
            audio = audio.float() / audio.float().abs().max()
            torchaudio.save(data_path, audio, 22050)
        results.append(data_path)
    return results


def generate_xtts(device):
    xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    results = []
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    val = ds.select(range(1000))
    val_other = ds.select(range(1000, len(ds)))
    # unique speakers
    speakers = set()
    speaker_dict = {}
    for item in val:
        speakers.add(item["speaker"])
    np.random.seed(42)
    Path("data/xtts").mkdir(exist_ok=True, parents=True)
    for speaker_id in speakers:
        # choose one wav per speaker from the other set
        audio_path = None
        for item in val_other:
            if item["speaker"] == speaker_id:
                audio_path = item["audio"]
                break
        if audio_path is None:
            # choose randomly
            audio_path = val_other[np.random.randint(len(val_other))]["audio"]
        speaker_dict[speaker_id] = audio_path
    for item in tqdm(val, "generating xtts"):
        data_path = Path("data/xtts") / Path(item["audio"]).name
        if not data_path.exists():
            xtts.tts_to_file(
                text=item["text"],
                speaker_wav=str(speaker_dict[item["speaker"]]),
                language="en",
                file_path=str(data_path),
            )
        results.append(data_path)
    return results