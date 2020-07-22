"""Dataset module."""
from __future__ import annotations
import dataclasses
from itertools import chain
from glob import glob
import os
import re
import tarfile
from typing import Any, Dict, Iterator, List

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow.compat.v1 as tf


# calculated by `$ python3 ./dataset.py`
STATS = {
    "num_examples": 12140,
    'num_stroke_frames': 7611737,
    'max_stroke_len': 1940,
    'max_text_len': 64,
    'mean': np.array(
        [3.52694747e+03, 3.27328457e+03, 1.37348168e+07], dtype=np.float32),
    'stddev': np.array(
        [2.71751375e+04, 2.54106321e+04, 1.01035389e+08], dtype=np.float32),
    'num_vocab': 81,
    'vocab': {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '?': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, '[': 53, ']': 54, 'a': 55, 'b': 56, 'c': 57, 'd': 58, 'e': 59, 'f': 60, 'g': 61, 'h': 62, 'i': 63, 'j': 64, 'k': 65, 'l': 66, 'm': 67, 'n': 68, 'o': 69, 'p': 70, 'q': 71, 'r': 72, 's': 73, 't': 74, 'u': 75, 'v': 76, 'w': 77, 'x': 78, 'y': 79, 'z': 80}  # noqa: E501
}


@dataclasses.dataclass
class ExamplePath:
    """Path struct to load Example."""
    text: str
    strokes: List[str]
    prefix: str


@dataclasses.dataclass
class RawExample:
    """Raw example struct."""
    text: str
    # list of (seqlen, 3)
    strokes: List[np.ndarray]


@dataclasses.dataclass
class TensorExample:
    """Example for tf.data.Dataset."""

    text: str
    text_ids: np.ndarray
    strokes: np.ndarray
    raw_strokes: np.ndarray
    strokes_lengths: np.ndarray
    strokes_length: int
    text_length: int
    end_flags: np.ndarray
    strokes_weight: np.ndarray
    text_weight: np.ndarray

    types = {
        "text": tf.string,
        "text_ids": tf.int32,
        "strokes": tf.float32,
        "raw_strokes": tf.float32,
        "end_flags": tf.float32,
        "strokes_lengths": tf.int32,
        "strokes_length": tf.int32,
        "strokes_weight": tf.float32,
        "text_length": tf.int32,
        "text_weight": tf.float32,
    }

    shapes = {
        "text": tf.TensorShape([]),
        "text_ids": tf.TensorShape([None]),
        "strokes": tf.TensorShape([None, 3]),
        "raw_strokes": tf.TensorShape([None, 3]),
        "end_flags": tf.TensorShape([None]),
        "strokes_lengths": tf.TensorShape([None]),
        "strokes_length": tf.TensorShape([]),
        "strokes_weight": tf.TensorShape([None]),
        "text_length": tf.TensorShape([]),
        "text_weight": tf.TensorShape([None]),
    }

    # TODO
    padding_values = {
        "end_flags": 1,
    }

    @classmethod
    def text_to_ids(cls, text: str) -> np.ndarray:
        """Convert text to an ascii-id array."""
        return np.array([STATS["vocab"][c] for c in text], dtype=np.int32)

    @classmethod
    def from_raw(cls, x: RawExample) -> TensorExample:
        """Load a raw example."""
        raw_strokes = np.concatenate(x.strokes)
        strokes = (raw_strokes - STATS["mean"]) / STATS["stddev"]
        end_flags = np.zeros(len(strokes), dtype=np.float32)
        strokes_lengths = np.array([len(s) for s in x.strokes], dtype=np.int32)
        t = 0
        for sl in strokes_lengths:
            t += sl
            end_flags[t-1] = 1
        return TensorExample(
            text=x.text,
            text_ids=cls.text_to_ids(x.text),
            strokes=strokes,
            raw_strokes=raw_strokes,
            end_flags=end_flags,
            strokes_lengths=strokes_lengths,
            strokes_length=sum(strokes_lengths),
            strokes_weight=np.ones(len(strokes), dtype=np.float32),
            text_weight=np.ones(len(x.text), dtype=np.float32),
            text_length=len(x.text),
        )


def calc_stats(xs: Iterator[RawExample]) -> Dict[str, Any]:
    """Calculate statistics over dataset."""
    # TODO: calc vocab
    n = 0
    num_examples = 0
    mean = np.zeros(3, dtype=np.float32)
    mean2 = np.zeros(3, dtype=np.float32)
    max_stroke_len = 0
    max_text_len = 0
    vocab = set()
    for x in xs:
        num_examples += 1
        max_text_len = max(max_text_len, len(x.text))
        max_stroke_len = max(max_stroke_len, sum(len(s) for s in x.strokes))
        for t in x.text:
            vocab.add(t)
        for s in x.strokes:
            m = s.shape[0]
            nm = n + m
            mean = n / nm * mean + s.sum(axis=0) / nm
            mean2 = n / nm * mean2 + s.sum(axis=0) ** 2 / nm
            n += m
    vocab_dict = {x: i for i, x in enumerate(sorted(list(vocab)))}
    return {
        "num_examples": num_examples,
        "num_stroke_frames": n,
        "max_stroke_len": max_stroke_len,
        "max_text_len": max_text_len,
        "mean": mean,
        "stddev": (mean2 - mean ** 2) ** 0.5,
        "num_vocab": len(vocab),
        "vocab": vocab_dict
    }


def extract_quoted(text: str) -> List[str]:
    """Extract quoted values in str."""
    return re.findall(r'\"(.+?)\"', text)


def load_strokes(path: str) -> List[np.ndarray]:
    """Load Stroke values from a file path."""
    ret = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("<Stroke "):
                xs: List[int] = []
                ys: List[int] = []
                ts: List[float] = []
            if line == "</Stroke>":
                ret.append(np.array([xs, ys, ts], dtype=np.float32).T)
            if line.startswith("<Point "):
                x, y, t = extract_quoted(line)
                xs.append(int(x))
                ys.append(int(y))
                ts.append(float(t))
    return ret


def plot_strokes(strokes: List[np.ndarray]) -> plt.Figure:
    """Plot strokes in a new pyplot figure."""
    fig = plt.figure()
    gca = fig.gca()
    # set x/y axis to equal scales
    gca.set_aspect('equal', adjustable='box')
    # low val to top, high to bottom
    gca.invert_yaxis()
    ax = fig.add_subplot()
    for s in strokes:
        ax.plot(s[0], s[1])
    return fig


def load_text(path: str) -> List[str]:
    """"Load text lines from a file path."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "CSR:":
                f.readline()
                break
        return [x.strip() for x in f]


def iter_example_path(root: str) -> Iterator[ExamplePath]:
    """Iterate example paths in root."""
    for text in glob(f"{root}/ascii/**/*.txt", recursive=True):
        # omit .txt
        prefix = text[:-4].replace(f"{root}/ascii", f"{root}/lineStrokes")
        strokes = glob(f"{prefix}*.xml")
        if strokes == []:
            continue
        yield ExamplePath(text=text, strokes=sorted(strokes), prefix=prefix)


def load_raw_examples(paths: ExamplePath) -> Iterator[RawExample]:
    """Load examples in paths."""
    text = load_text(paths.text)
    strokes = [load_strokes(s) for s in paths.strokes]
    if len(text) != len(strokes):
        tf.logging.warning(f"length mismatch in files: {paths.prefix}")
        return
    for t, s in zip(text, strokes):
        yield RawExample(text=t, strokes=s)


def load_dataset(root: str) -> Iterator[RawExample]:
    """Load dataset in root."""
    return chain.from_iterable(map(load_raw_examples, iter_example_path(root)))


def load_tf_dataset(root: str) -> tf.data.Dataset:
    """Load tf.data.Dataset."""
    def gen():
        for raw in load_dataset(root):
            yield dataclasses.asdict(TensorExample.from_raw(raw))
    return tf.data.Dataset.from_generator(
        generator=gen,
        output_types=TensorExample.types,
        output_shapes=TensorExample.shapes,
    )


def download_tgz(username: str, password: str, root: str):
    """Download required tgz."""
    os.makedirs(root, exist_ok=True)
    for fname in ["ascii-all.tar.gz", "lineStrokes-all.tar.gz"]:
        dst = os.path.join(root, fname)
        if os.path.exists(dst):
            tf.logging.info(f"Found {dst}.")
            continue

        if username == "" or password == "":
            raise ValueError(
                "Register username and password at http://www.fki.inf.unibe.ch"
                "/databases/iam-on-line-handwriting-database")

        tf.logging.info(f"Downloading {fname}.")
        r = requests.get(
            f"http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/{fname}",
            auth=(username, password))
        r.raise_for_status()

        # process tarfile
        cwd = os.getcwd()
        os.chdir(root)
        with open(fname, "wb") as f:
            f.write(r.content)
        with tarfile.open(fname, "r") as t:
            t.extractall()
        os.chdir(cwd)


if __name__ == "__main__":
    STATS = calc_stats(load_dataset("data"))
    print(STATS)
