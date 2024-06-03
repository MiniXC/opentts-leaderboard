from embedding_models import (
    MFCC,
    Hubert, 
    Wav2Vec2, 
    DVector,
    DVectorIntra, 
    XVector, 
    XVectorIntra,
    Voicefixer, 
    Whisper, 
    Wav2Vec2WER, 
    ProsodyMPM,
    AllosaurusPhone,
)
from kaldi_wer import get_kaldi_wasserstein
from hubert_units import HubertUnitCounter
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from contextlib import contextmanager
from rich.console import Console

console = Console()

@contextmanager
def suppress_stdout():
    """A context manager to suppress all stdout output."""
    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass

    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = save_stdout

class MeasureChecker:
    def __init__(self, df, overwrite=False):
        self.df = df
        self.overwrite = overwrite

    def check_if_measure_and_source_exists(self, measure, source):
        if self.overwrite:
            return False
        if self.df.empty:
            return False
        vals = self.df.loc[(self.df["source"] == source) & (self.df["measure"] == measure)]["value"].values
        return len(vals) >= 1 and isinstance(vals[0], np.float64)

    def check_if_measure_exists_for_all_sources(self, measure, sources):
        return all(self.check_if_measure_and_source_exists(measure, source) for source in sources)

    def add_to_df(self, measure, source, value):
        if self.check_if_measure_and_source_exists(measure, source):
            return self.df
        new_entry = pd.DataFrame({
            "source": [source],
            "measure": [measure],
            "value": [value]
        })
        self.df = pd.concat([self.df, new_entry], ignore_index=True)
        self.df.to_csv("results/data.csv", index=False)
        return self.df

class MeasureProcessor:
    def __init__(self, sources, measure_checker, overwrite=False):
        self.sources = sources
        self.measure_checker = measure_checker
        self.overwrite = overwrite
        measure_checker.overwrite = overwrite

    def process_measure(self, console, measure_name, model_class):
        console.rule(measure_name)
        if not self.measure_checker.check_if_measure_exists_for_all_sources(measure_name, self.sources) or self.overwrite:
            if self.overwrite:
                console.print(f"[bold yellow]Overwriting {measure_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]{measure_name} not found in cache. Computing...[/bold yellow]")
            model = model_class()
            for source in self.sources:
                if not self.measure_checker.check_if_measure_and_source_exists(measure_name, source):
                    console.print(f"[yellow]Processing {source} for {measure_name}[/yellow]")
                    if hasattr(model, "get_frechet"):
                        self.measure_checker.add_to_df(measure_name, source, model.get_frechet(source, self.overwrite))
                    elif hasattr(model, "get_wasserstein"):
                        self.measure_checker.add_to_df(measure_name, source, model.get_wasserstein(source, self.overwrite))
        else:
            console.print(f"[bold green]{measure_name} found in cache.[/bold green]")

    def process_hubert_units(self, console):
        measure_lengths = "General/Hubert Units/Lengths"
        measure_counts = "General/Hubert Units/Counts"
        console.rule("General/Hubert Units")
        if (not self.measure_checker.check_if_measure_exists_for_all_sources(measure_lengths, self.sources) and \
           not self.measure_checker.check_if_measure_exists_for_all_sources(measure_counts, self.sources)) or self.overwrite:
            if self.overwrite:
                console.print(f"[bold yellow]Overwriting {measure_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]Hubert Units not found in cache. Computing...[/bold yellow]")
            hubert_units = HubertUnitCounter()
            for source in self.sources:
                if not (self.measure_checker.check_if_measure_and_source_exists("General/Hubert Units/Lengths", source) and \
                     self.measure_checker.check_if_measure_and_source_exists("General/Hubert Units/Counts", source)):
                    console.print(f"[yellow]Processing {source} for General/Hubert Units[/yellow]")
                    lengths, counts = hubert_units.get_wasserstein_distances(source)
                    self.measure_checker.add_to_df(measure_lengths, source, lengths)
                    self.measure_checker.add_to_df(measure_counts, source, counts)
        else:
            console.print(f"[bold green]Hubert Units found in cache.[/bold green]")

    def process_dvector_intra(self, console):
        measure_name = "Speaker/DVector/Intra"
        console.rule(measure_name)
        if not self.measure_checker.check_if_measure_exists_for_all_sources(measure_name, self.sources) or self.overwrite:
            if self.overwrite:
                console.print(f"[bold yellow]Overwriting {measure_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]{measure_name} not found in cache. Computing...[/bold yellow]")
            for source in self.sources:
                if not self.measure_checker.check_if_measure_and_source_exists("Speaker/DVector/Intra", source):
                    console.print(f"[yellow]Processing {source} for {measure_name}[/yellow]")
                    dvector = DVectorIntra(source)
                    self.measure_checker.add_to_df(measure_name, source, dvector.get_frechet(source))
        else:
            console.print(f"[bold green]{measure_name} found in cache.[/bold green]")

    def process_xvector_intra(self, console):
        measure_name = "Speaker/XVector/Intra"
        console.rule(measure_name)
        if not self.measure_checker.check_if_measure_exists_for_all_sources(measure_name, self.sources) or self.overwrite:
            if self.overwrite:
                console.print(f"[bold yellow]Overwriting {measure_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]{measure_name} not found in cache. Computing...[/bold yellow]")
            for source in self.sources:
                if not self.measure_checker.check_if_measure_and_source_exists("Speaker/XVector/Intra", source):
                    console.print(f"[yellow]Processing {source} for {measure_name}[/yellow]")
                    xvector = XVectorIntra(source)
                    self.measure_checker.add_to_df(measure_name, source, xvector.get_frechet(source))
        else:
            console.print(f"[bold green]{measure_name} found in cache.[/bold green]")

    def process_kaldi_asr(self, console):
        measure_name = "Training/Kaldi ASR"
        console.rule(measure_name)
        if not self.measure_checker.check_if_measure_exists_for_all_sources(measure_name, self.sources) or self.overwrite:
            if self.overwrite:
                console.print(f"[bold yellow]Overwriting {measure_name}...[/bold yellow]")
            else:
                console.print(f"[bold yellow]{measure_name} not found in cache. Computing...[/bold yellow]")
            for source in self.sources:
                if not self.measure_checker.check_if_measure_and_source_exists("Training/Kaldi ASR", source):
                    console.print(f"[yellow]Processing {source} for {measure_name}[/yellow]")
                    self.measure_checker.add_to_df(measure_name, source, get_kaldi_wasserstein(source))
        else:
            console.print(f"[bold green]{measure_name} found in cache.[/bold green]")

if __name__ == "__main__":
    Path("cache").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if not Path("results/data.csv").exists():
        df = pd.DataFrame(columns=["source", "measure", "value"])
    else:
        df = pd.read_csv("results/data.csv")

    sources = ["reference.dev", "amphion_fastspeech2", "parler", "hifigan", "xtts", "ljspeech", "lj_tacotron2"]

    measure_checker = MeasureChecker(df)
    measure_processor = MeasureProcessor(sources, measure_checker, overwrite=False)

    measure_processor.process_measure(console, "General/Allosaurus", AllosaurusPhone)
    measure_processor.process_measure(console, "Environment/Voicefixer", Voicefixer)
    measure_processor.process_measure(console, "General/MFCC", MFCC)
    measure_processor.process_measure(console, "General/Hubert", Hubert)
    measure_processor.process_measure(console, "General/Wav2Vec2", Wav2Vec2)
    measure_processor.process_hubert_units(console)
    measure_processor.process_measure(console, "Prosody/MPM", ProsodyMPM)
    measure_processor.process_measure(console, "Speaker/DVector/General", DVector)
    measure_processor.process_dvector_intra(console)
    measure_processor.process_measure(console, "Speaker/XVector/General", XVector)
    measure_processor.process_xvector_intra(console)
    measure_processor.process_measure(console, "Environment/Voicefixer", Voicefixer)
    measure_processor.process_measure(console, "Intelligibility/Whisper", Whisper)
    measure_processor.process_measure(console, "Intelligibility/Wav2Vec2", Wav2Vec2WER)
    measure_processor.process_kaldi_asr(console)
