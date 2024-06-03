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
from subprocess import run as _run

N_SAMPLES = 1000
DEV_SUBSET = 100

def run(cmd):
    return _run(cmd, shell=True)

def generate_data(train_ds, device="cuda"):
    if not Path("data").exists():
        Path("data").mkdir()
    if train_ds in ["ref_test", "reference.test"] or "reference.test" in train_ds:
        return generate_ref_test(train_ds.replace("reference.test", "").replace(".", ""))
    if train_ds in ["ref_dev", "reference.dev"]:
        return generate_ref_dev()
    if train_ds == "parler":
        return generate_parler_tts(device)
    if train_ds == "hifigan":
        return generate_hifigan()
    if train_ds == "xtts":
        return generate_xtts(device)
    if train_ds == "ljspeech":
        return generate_ljspeech_ref()
    if train_ds == "lj_tacotron2":
        return generate_tacotron(device)
    if train_ds == "amphion_fastspeech2":
        return generate_amphion_fastspeech2()
    if train_ds == "noise":
        return generate_noise()
    raise ValueError(f"Unknown dataset {train_ds}")

def generate_noise():
    noise = ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    noise = noise.shuffle(seed=42)
    noise = noise.select(range(N_SAMPLES))
    if DEV_SUBSET:
        noise = noise.select(range(DEV_SUBSET))
    Path("data/noise").mkdir(exist_ok=True, parents=True)
    results = []
    text_results = []
    speakers = []
    for item in tqdm(noise, "generating noise"):
        data_path = Path("data/noise") / Path(item["audio"]).name
        if not data_path.exists():
            _audio, _sr = torchaudio.load(item["audio"])
            audio = torch.randn_like(_audio)
            torchaudio.save(data_path, audio, _sr)
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write("noise")
        results.append(data_path)
        text_results.append("noise")
        speakers.append("noise")
    return results, text_results, speakers

def generate_ref_test(num):
    audios = []
    texts = []
    speakers = []
    if len(num) > 0:
        num = int(num)
        np.random.seed(num)
    if Path("data/ref_test").exists() and len(list(Path("data/ref_test").glob("*.wav"))) == N_SAMPLES:
        for audio_path in Path("data/ref_test").glob("*.wav"):
            audios.append(str(audio_path))
            speakers.append(audio_path.name.split("_")[0])
            with open(audio_path.with_suffix(".txt")) as f:
                texts.append(f.read())
        if DEV_SUBSET:
            # randomly sample
            indices = np.random.choice(range(len(audios)), DEV_SUBSET, replace=False)
            audios = [audios[i] for i in indices]
            texts = [texts[i] for i in indices]
            speakers = [speakers[i] for i in indices]
        return audios, texts, speakers
    ds = load_dataset(
        "cdminix/libritts-aligned", split="test.clean", trust_remote_code=True
    )
    ds = ds.shuffle()
    ds = ds.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ds = ds.select(range(DEV_SUBSET))
    audios, texts, speakers = (
        [item["audio"] for item in ds], 
        [item["text"] for item in ds],
        [item["speaker"] for item in ds],
    )
    # write to file
    Path("data/ref_test").mkdir(exist_ok=True, parents=True)
    for audio, text in zip(audios, texts):
        audio_path = Path("data/ref_test") / Path(audio).name
        if not audio_path.exists():
            audio, sr = torchaudio.load(audio)
            torchaudio.save(audio_path, audio, sr)
        with open(audio_path.with_suffix(".txt"), "w") as f:
            f.write(text)
    return audios, texts, speakers

def generate_ref_dev():
    audios = []
    texts = []
    speakers = []
    if Path("data/ref_dev").exists() and len(list(Path("data/ref_dev").glob("*.wav"))) == N_SAMPLES:
        for audio_path in Path("data/ref_dev").glob("*.wav"):
            audios.append(str(audio_path))
            speakers.append(audio_path.name.split("_")[0])
            with open(audio_path.with_suffix(".txt")) as f:
                texts.append(f.read())
        if DEV_SUBSET:
            # randomly sample
            np.random.seed(42)
            indices = np.random.choice(range(len(audios)), DEV_SUBSET, replace=False)
            audios = [audios[i] for i in indices]
            texts = [texts[i] for i in indices]
            speakers = [speakers[i] for i in indices]
        return audios, texts, speakers
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ds = ds.select(range(DEV_SUBSET))
    audios, texts, speakers = (
        [item["audio"] for item in ds], 
        [item["text"] for item in ds],
        [item["speaker"] for item in ds],
    )
    # write to file
    Path("data/ref_dev").mkdir(exist_ok=True, parents=True)
    for audio, text in zip(audios, texts):
        audio_path = Path("data/ref_dev") / Path(audio).name
        if not audio_path.exists():
            audio, sr = torchaudio.load(audio)
            torchaudio.save(audio_path, audio, sr)
        with open(audio_path.with_suffix(".txt"), "w") as f:
            f.write(text)
    return audios, texts, speakers

def generate_parler_tts(device="cuda"):
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    parler_val = load_dataset("parler-tts/libritts_r_tags_tagged_10k_generated", "clean", split="test.clean")
    parler_val = parler_val.shuffle(seed=42)
    parler_val = parler_val.select(range(N_SAMPLES))
    if DEV_SUBSET:
        parler_val = parler_val.select(range(DEV_SUBSET))
    Path("data/parler").mkdir(exist_ok=True, parents=True)
    speaker_dict = {}
    results = []
    text_results = []
    speakers = []
    for item in tqdm(parler_val, "generating parler tts"):
        if Path(f"data/parler/{item['id']}.wav").exists():
            results.append(f"data/parler/{item['id']}.wav")
            text_results.append(item["text"])
            speakers.append(item["speaker_id"])
            continue
        prompt_input_ids = parler_tokenizer(item["text"], return_tensors="pt").input_ids.to(device)
        # we only use one description per speaker
        if item["speaker_id"] not in speaker_dict:
            input_ids = parler_tokenizer(item["text_description"], return_tensors="pt").input_ids
            speaker_dict[item["speaker_id"]] = input_ids
        input_ids = speaker_dict[item["speaker_id"]].to(device)
        with torch.no_grad():
            generated = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            if parler_model.config.sampling_rate != 16000:
                generated = torchaudio.transforms.Resample(parler_model.config.sampling_rate, 16000)(generated.to("cpu").float())
        generated = generated / generated.abs().max()
        torchaudio.save(f"data/parler/{item['id']}.wav", generated, 16000)
        with open(f"data/parler/{item['id']}.txt", "w") as f:
            f.write(item["text"])
        results.append(f"data/parler/{item['id']}.wav")
        text_results.append(item["text"])
        speakers.append(item["speaker_id"])
    return results, text_results, speakers

def generate_hifigan():
    hifigan = Synthesiser()
    results = []
    text_results = []
    speakers = []
    val = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    val = val.shuffle(seed=42)
    val = val.select(range(N_SAMPLES))
    if DEV_SUBSET:
        val = val.select(range(DEV_SUBSET))
    for audio in tqdm(val, "generating hifigan"):
        text = audio["text"]
        audio_path = audio["audio"]
        speakers.append(audio["speaker"])
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
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(text)
        results.append(data_path)
        text_results.append(text)
    return results, text_results, speakers


def generate_xtts(device):
    xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    results = []
    text_results = []
    speakers_l = []
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    val = ds.select(range(N_SAMPLES))
    val_other = ds.select(range(N_SAMPLES, len(ds)))
    if DEV_SUBSET:
        val = ds.select(range(DEV_SUBSET))
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
        speakers_l.append(item["speaker"])
        if not data_path.exists():
            xtts.tts_to_file(
                text=item["text"],
                speaker_wav=str(speaker_dict[item["speaker"]]),
                language="en",
                file_path=str(data_path),
            )
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(item["text"])
        results.append(data_path)
        text_results.append(item["text"])
    return results, text_results, speakers_l

def generate_ljspeech_ref():
    ljspeech = load_dataset("lj_speech", split="train")
    ljspeech = ljspeech.shuffle(seed=42)
    ljspeech = ljspeech.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ljspeech = ljspeech.select(range(DEV_SUBSET))
    Path("data/ljspeech").mkdir(exist_ok=True, parents=True)
    results = []
    text_results = []
    speakers = []
    for item in tqdm(ljspeech, "generating ljspeech"):
        data_path = Path("data/ljspeech") / Path(item["file"]).name.replace("-", "_")
        if not data_path.exists():
            audio = torch.tensor(item["audio"]["array"]).float().unsqueeze(0)
            torchaudio.save(data_path, audio, 22050)
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(item["text"])
        results.append(data_path)
        text_results.append(item["text"])
        speakers.append("ljspeech")
    return results, text_results, speakers


def generate_tacotron(device):
    Path("data/ljspeech_tacotron").mkdir(exist_ok=True, parents=True)
    results = []
    text_results = []
    speakers = []
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ds = ds.select(range(DEV_SUBSET))
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    for item in tqdm(ds, "generating ljspeech tacotron"):
        data_path = Path("data/ljspeech_tacotron") / Path(item["audio"]).name
        if not data_path.exists():
            tts.tts_to_file(
                text=item["text"],
                file_path=str(data_path),
            )
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(item["text"])
        results.append(data_path)
        text_results.append(item["text"])
        speakers.append("ljspeech")
    return results, text_results, speakers

def generate_amphion_fastspeech2():
    Path("data/amphion_fastspeech2").mkdir(exist_ok=True, parents=True)
    Path("amphion").mkdir(exist_ok=True)
    results = []
    text_results = []
    speakers = []
    if not Path("amphion/amphion_fastspeech2").exists():
        run("git lfs install")
        run("git clone https://huggingface.co/amphion/fastspeech2_ljspeech amphion/amphion_fastspeech2")
    if not Path("amphion/amphion_hifigan").exists():
        run("git clone https://huggingface.co/amphion/hifigan_ljspeech amphion/amphion_hifigan")
        run("cp amphion/amphion_hifigan/args.json amphion/amphion_hifigan/checkpoints/args.json")
    if not Path("amphion/amphion").exists():
        run("git clone https://github.com/open-mmlab/Amphion.git amphion/amphion")
    run("rm -rf amphion/amphion/ckpts/tts")
    run("mkdir -p amphion/amphion/ckpts/tts")
    abs_path = Path("amphion/amphion_fastspeech2").resolve()
    abs_path_hifigan = Path("amphion/amphion_hifigan").resolve()
    run(f"ln -s {abs_path} amphion/amphion/ckpts/tts/")
    run(f"rm -rf amphion/amphion/ckpts/vocoder")
    run(f"mkdir -p amphion/amphion/ckpts/vocoder")
    run(f"ln -s {abs_path_hifigan} amphion/amphion/ckpts/vocoder/")
    if not Path("amphion/amphion/ckpts/tts/fastspeech").exists():
        run("mkdir -p amphion/amphion/ckpts/tts/fastspeech")
        run(f"ln -s {abs_path}/LJSpeech amphion/amphion/ckpts/tts/fastspeech/")
    amphion_dir = Path("amphion/amphion").resolve()
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ds = ds.select(range(DEV_SUBSET))
    for item in tqdm(ds, "generating amphion fastspeech2"):
        data_path = Path("data/amphion_fastspeech2") / Path(item["audio"]).name
        if not data_path.exists():
            run(f'sh {amphion_dir}/egs/tts/FastSpeech2/run.sh --stage 3 \
            --config {amphion_dir}/ckpts/tts/amphion_fastspeech2/args.json \
            --infer_expt_dir {amphion_dir}/ckpts/tts/amphion_fastspeech2/ \
            --infer_output_dir {amphion_dir}/ckpts/tts/amphion_fastspeech2/results \
            --infer_mode single \
            --infer_text "{item["text"]}" \
            --vocoder_dir {amphion_dir}/ckpts/vocoder/amphion_hifigan/checkpoints/ \
            ')
            run(f"mv {amphion_dir}/ckpts/tts/amphion_fastspeech2/results/single/test_pred.wav {data_path}")
            run(f"rm -rf {amphion_dir}/ckpts/tts/amphion_fastspeech2/results/single")
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(item["text"])
        results.append(data_path)
        text_results.append(item["text"])
        speakers.append("ljspeech")
    return results, text_results, speakers

def generate_valle():
    Path("data/valle").mkdir(exist_ok=True, parents=True)
    results = []
    text_results = []
    speakers = []
    if not Path("amphion/amphion_valle").exists():
        run("git lfs install")
        run("git clone https://huggingface.co/amphion/fastspeech2_ljspeech amphion/amphion_valle")
    if not Path("amphion/amphion").exists():
        run("git clone https://github.com/open-mmlab/Amphion.git amphion/amphion")
    run("rm -rf amphion/amphion/ckpts/tts")
    run("mkdir -p amphion/amphion/ckpts/tts")
    abs_path = Path("amphion/amphion_valle").resolve()
    run(f"ln -s {abs_path} amphion/amphion/ckpts/tts/")
    # TODO: finish using https://huggingface.co/amphion/valle_libritts
    ds = load_dataset(
        "cdminix/libritts-aligned", split="dev.clean", trust_remote_code=True
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(N_SAMPLES))
    if DEV_SUBSET:
        ds = ds.select(range(DEV_SUBSET))
    for item in tqdm(ds, "generating valle"):
        data_path = Path("data/valle") / Path(item["audio"]).name
        if not data_path.exists():
            run(f"python valle/valle.py --text '{item['text']}' --output {data_path}")
            with open(data_path.with_suffix(".txt"), "w") as f:
                f.write(item["text"])
        results.append(data_path)
        text_results.append(item["text"])
        speakers.append("ljspeech")
    return results, text_results, speakers