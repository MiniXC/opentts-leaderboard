from pathlib import Path
from dataclasses import dataclass, fields
import subprocess
import multiprocessing
from typing import List
import os
from multiprocessing import cpu_count
import re
import shutil
from time import sleep
import sys
import selectors
from glob import glob
from dotenv import load_dotenv
from shutil import copy

cpus = cpu_count()

load_dotenv()

KALDI_PATH = os.environ.get("KALDI_PATH", "/kaldi")

@dataclass
class Args:
    kaldi_path: str = f"{KALDI_PATH}/egs/librispeech/s5"
    data_path: str = "kaldi_data/"
    data_name: str = "LibriSpeech"
    lm_path: str = "{data}/local/lm"
    dict_path: str = "{data}/local/dict"
    dict_nosp_path: str = "{data}/local/dict_nosp"
    lang_path: str = "{data}/lang"
    lang_tmp_path: str = "{data}/local/lang_tmp"
    lang_nosp_path: str = "{data}/lang_nosp"
    lang_nosp_tmp_path: str = "{data}/local/lang_tmp_nosp"
    lm_url: str = "www.openslr.org/resources/11"
    log_file: str = "train.log"
    test_sets: str = "dev_clean,test_clean"
    local_data_path: str = "data"
    local_exp_path: str = "exp"
    mfcc_path: str = "{data}/mfcc"
    train_name: str = "synth"
    train_set: str = "n_topline"
    train_cmd: str = "run.pl"
    lm_names: str = "tgsmall,tgmed"  # tglarge,fglarge
    mfcc_path: str = "{exp}/make_mfcc"
    mono_subset: int = 2000
    mono_path: str = "{exp}/mono"
    tri1_subset: int = 4000
    tri1_path: str = "{exp}/tri1"
    tri2b_path: str = "{exp}/tri2b"
    tri3b_path: str = "{exp}/tri3b"
    log_stages: str = "all"
    tdnn_path: str = "{exp}/tdnn"
    verbose: bool = True
    run_name: str = None
    clean_stages: str = "none"
    use_cmvn: bool = False
    use_cnn: bool = False

args = Args()

def prepare_data():
    for directory in Path("data").iterdir():
        if directory.is_dir():
            target_dir = Path(args.data_path) / directory.name
            for wav in tqdm(list(directory.glob("*.wav"))):
                speaker_dir = target_dir / wav.name.split("_")[0]
                speaker_dir.mkdir(exist_ok=True)
                txt = str(wav).replace(".wav", ".txt")
                copy(txt, target / wav.parent.name / Path(txt).name)
                copy(wav, speaker_dir / wav.name)

class Tasks:
    def __init__(self, logfile, kaldi_path):
        self.logfile = logfile
        self.kaldi_path = kaldi_path

    def execute(self, command, **kwargs):
        p = subprocess.Popen(
            f"{command}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )

        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        break_loop = False

        while not break_loop:
            for key, _ in sel.select():
                data = key.fileobj.read1().decode()
                if not data:
                    break_loop = True
                    break
                if key.fileobj is p.stdout:
                    yield data
                else:
                    yield data

        p.stdout.close()
        return_code = p.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def run(
        self,
        command,
        check_path=None,
        desc=None,
        run_in_kaldi=True,
        run_in=None,
        clean=False,
    ):
        if check_path is not None:
            if isinstance(check_path, list):
                run_command = any([not Path(p).exists() for p in check_path])
            else:
                run_command = not Path(check_path).exists()
        else:
            run_command = True
        if clean:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for c_p in check_path:
                c_p = Path(c_p)
                if c_p.is_dir():
                    shutil.rmtree(c_p)
                if c_p.is_file():
                    os.remove(c_p)
            run_command = True
        if run_command:
            if desc is None:
                desc = "[bright_black]" + command + "[/bright_black]"
            with console.status(desc):
                if run_in_kaldi:
                    for path in self.execute(command, cwd=self.kaldi_path):
                        if args.verbose:
                            print(path, end="")
                elif run_in is None:
                    for path in self.execute(command):
                        if args.verbose:
                            print(path, end="")
                else:
                    for path in self.execute(command, cwd=run_in):
                        if args.verbose:
                            print(path, end="")
            print(f"[green]✓[/green] {desc}")
        else:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for p in check_path:
                print(f"[blue]{p} already exists[blue]")


def score_model(task, args, path, name, fmllr=False, lang_nosp=True):
    global run_name
    if args.run_name is not None:
        run_name = args.run_name
    if args.log_stages == "all" or name in args.log_stages.split(","):
        print(
            {
                "model": name,
                "group_name": run_name,
                "train_set": args.train_set
            }
        )
        mkgraph_args = ""
        if lang_nosp:
            graph_path = path / "graph_tgsmall"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        else:
            graph_path = path / "graph_tgsmall_sp"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        for i, tst in enumerate(args.test_sets):
            graph_test_path = str(graph_path) + f"_{tst}"
            tst_path = args.local_data_path / (tst + "_small")
            # tst_path = args.local_data_path / (tst + "_med")
            if fmllr:
                p_decode = "_fmllr"
            else:
                p_decode = ""
            task.run(
                f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
                graph_test_path,
            )
            scoring_path = Path(graph_test_path) / "scoring_kaldi"
            task.run(
                f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
                scoring_path,
            )
            with open(scoring_path / "best_wer", "r") as best_wer:
                best_wer = best_wer.read()
            wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
            ins_err = int(re.findall(r"(\d+) ins", best_wer)[0])
            del_err = int(re.findall(r"(\d+) del", best_wer)[0])
            sub_err = int(re.findall(r"(\d+) sub", best_wer)[0])
            with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
                bootci = bootci.read()
            lower_wer, upper_wer = [
                round(float(c), 2)
                for c in re.findall(
                    r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci
                )[0]
            ]
            ci = round((upper_wer - lower_wer) / 2, 2)
            print(
                {
                    f"{tst}/wer": wer,
                    f"{tst}/wer_lower": lower_wer,
                    f"{tst}/wer_upper": upper_wer,
                    f"{tst}/ci_width": ci,
                    f"{tst}/ins": ins_err,
                    f"{tst}/del": del_err,
                    f"{tst}/sub": sub_err,
                }
            )



def run(args, train_ds, test_ds):
    task = Tasks(args.log_file, args.kaldi_path)
    args.test_sets = args.test_sets.split(",")
    args.lm_names = args.lm_names.split(",")
    if "," in args.clean_stages:
        args.clean_stages = args.clean_stages.split(",")
    else:
        args.clean_stages = [args.clean_stages]

    for field in fields(args):
        k, v = field.name, getattr(args, field.name)
        if "path" in field.name:
            if "{data}" in v:
                v = v.replace("{data}", str(args.local_data_path))
            if "{exp}" in v:
                v = v.replace("{exp}", str(args.local_exp_path))
            setattr(args, field.name, Path(v).resolve())

    # download lm
    print(f"local/download_lm.sh {args.lm_url} {args.lm_path}")
    task.run(f"local/download_lm.sh {args.lm_url} {args.lm_path}", args.lm_path)

    # prep train data
    train_path = Path(args.data_path) / args.train_name
    dest_path = Path(args.local_data_path) / args.train_set
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
        load_synth(train_path, dest_path)
        print(f"loading synthetic train data")
    else:
        print(f"{dest_path} already exists")

    args.train_set = args.train_set.replace("/", "_")

    # prep test data
    for tst in args.test_sets:
        tst_path = Path(args.data_path) / args.data_name / tst.replace("_", "-")
        dest_path = Path(args.local_data_path) / tst
        task.run(f"local/data_prep.sh {tst_path} {dest_path}", dest_path)
        if (Path(dest_path) / "spk2gender").exists():
            task.run(f"rm {str(Path(dest_path) / 'spk2gender')}", run_in_kaldi=False)

    # create lms
    task.run(
        f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
        args.dict_nosp_path,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
        args.lang_nosp_tmp_path,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}",
        [
            str(args.lang_nosp_path) + "_test_tgsmall",
            str(args.lang_nosp_path) + "_test_tgmed",
        ],
    )
    if "tgsmall" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False
        )
    if "tgmed" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False
        )
    if "tglarge" in args.lm_names:
        tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}",
            tglarge_path,
        )
    if "fglarge" in args.lm_names:
        fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}",
            fglarge_path,
        )

    # mfccs
    for tst in args.test_sets + [args.train_set]:
        tst_path = args.local_data_path / tst
        exp_path = args.mfcc_path / tst
        task.run(
            f"steps/make_mfcc.sh --cmd {args.train_cmd} --nj {cpus} {tst_path} {exp_path} mfccs",
            tst_path / "feats.scp",
        )
        task.run(
            f"steps/compute_cmvn_stats.sh {tst_path} {exp_path} mfccs",
            exp_path / f"cmvn_{tst}.log",
        )

    # generate subsets
    for tst in args.test_sets:
        dest_path = Path(args.local_data_path) / tst
        task.run(
            f'utils/subset_data_dir.sh {dest_path} 100 {str(dest_path) + "_small"}',
            str(dest_path) + "_small",
        )

    train_path = args.local_data_path / args.train_set

    # mono
    mono_data_path = str(train_path) + "_mono"
    clean_mono = "mono" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"utils/subset_data_dir.sh --shortest {train_path} {args.mono_subset} {mono_data_path}",
        mono_data_path,
    )
    task.run(
        f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}",
        args.mono_path,
        clean=clean_mono,
    )
    score_model(task, args, args.mono_path, "mono")

    # tri1
    tri1_data_path = str(train_path) + "_tri1"
    clean_tri1 = "tri1" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"utils/subset_data_dir.sh {train_path} {args.tri1_subset} {tri1_data_path}",
        tri1_data_path,
    )
    tri1_ali_path = str(args.tri1_path) + "_ali"
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}",
        tri1_ali_path,
        clean=clean_tri1,
    )
    task.run(
        f"steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}",
        args.tri1_path,
        clean=clean_tri1,
    )
    score_model(task, args, args.tri1_path, "tri1", True)

    # tri2b
    tri2b_ali_path = str(args.tri2b_path) + "_ali"
    clean_tri2b = "tri2b" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}",
        tri2b_ali_path,
        clean=clean_tri2b,
    )

    task.run(
        f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
        args.tri2b_path,
        clean=clean_tri2b,
    )
    score_model(task, args, args.tri2b_path, "tri2b", True)

    # tri3b
    tri3b_ali_path = str(args.tri3b_path) + "_ali"
    clean_tri3b = "tri3b" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}",
        tri3b_ali_path,
        clean=clean_tri3b,
    )
    task.run(
        f"steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}",
        args.tri3b_path,
        clean=clean_tri3b,
    )
    score_model(task, args, args.tri3b_path, "tri3b", True)

    # recompute lm
    task.run(
        f"steps/get_prons.sh --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri3b_path}",
        [
            args.tri3b_path / "pron_counts_nowb.txt",
            args.tri3b_path / "sil_counts_nowb.txt",
            args.tri3b_path / "pron_bigram_counts_nowb.txt",
        ],
        clean=clean_tri3b,
    )
    task.run(
        f'utils/dict_dir_add_pronprobs.sh --max-normalize true {args.dict_nosp_path} {args.tri3b_path / "pron_counts_nowb.txt"} {args.tri3b_path / "sil_counts_nowb.txt"} {args.tri3b_path / "pron_bigram_counts_nowb.txt"} {args.dict_path}',
        args.dict_path,
        clean=clean_tri3b,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_path} "<UNK>" {args.lang_tmp_path} {args.lang_path}',
        args.lang_path,
        clean=clean_tri3b,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_path} {args.lm_path}",
        [
            str(args.lang_path) + "_test_tgsmall",
            str(args.lang_path) + "_test_tgmed",
        ],
        clean=clean_tri3b,
    )
    score_model(task, args, args.tri3b_path, "tri3b-probs", True, False)


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    run(args)
