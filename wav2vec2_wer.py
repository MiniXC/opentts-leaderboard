from generate_data import generate_data
import re
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer
from torch.utils.data import DataLoader, Dataset
import torchaudio
from pathlib import Path
from datasets import load_metric
import os
import numpy as np
from jiwer import wer

from wav2vec2_collator import DataCollatorCTCWithPadding

os.environ["WANDB_DISABLED"] = "true"

Path("cache/wav2vec2").mkdir(exist_ok=True, parents=True)

class AudioTextDataset(Dataset):
    def __init__(self, audio, text, processor):
        self.audio = audio
        self.text = text
        self.processor = processor
    
    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.audio[idx])
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        audio = audio.squeeze(0)
        audio_input_values = self.processor(audio.numpy(), sampling_rate=16000).input_values
        with self.processor.as_target_processor():
            labels = self.processor(self.text[idx]).input_ids
        return {
            "input_values": audio_input_values,
            "labels": labels,
        }

def compute_metrics(pred):
    global wer_per_example
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    print(pred_str[:5], label_str[:5])

    wer_per_example = [wer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]

    print(wer_per_example[:5])
    return {
        "wer": np.mean(wer_per_example),
    }

def get_wav2vec2_wer(train_ds, test_ds, device="cuda"):
    global processor
    cache_path = Path("cache/wav2vec2") / f"wav2vec2_{train_ds}_{test_ds}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    train_audio, train_text = generate_data(train_ds)
    test_audio, test_text = generate_data(test_ds)
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    train_text = [re.sub(chars_to_ignore_regex, '', text).lower() for text in train_text]
    test_text = [re.sub(chars_to_ignore_regex, '', text).lower() for text in test_text]
    vocab = set()
    for text in train_text + test_text:
        vocab.update(list(text))
    vocab = sorted(list(vocab))
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("cache/wav2vec2/vocab.json", "w") as f:
        json.dump(vocab_dict, f)
    tokenizer = Wav2Vec2CTCTokenizer("cache/wav2vec2/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    train_ds = AudioTextDataset(train_audio, train_text, processor)
    test_ds = AudioTextDataset(test_audio, test_text, processor)
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()
    from transformers import TrainingArguments

    training_args = TrainingArguments(
    # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
        output_dir="cache/wav2vec2/train",
        group_by_length=True,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        eval_steps=200,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.005,
        warmup_steps=100,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    print(wer_per_example)
    np.save(cache_path, wer_per_example)
    return wer_per_example



get_wav2vec2_wer("reference.dev", "reference.test")