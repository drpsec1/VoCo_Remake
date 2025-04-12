#!/usr/bin/env python3
"""
Voice Cloner GUI
----------------
Simple PySide6 desktop interface that wraps a zero‑shot voice‑cloning pipeline.
The GUI lets you
  • Select one or more MP3/ WAV reference clips for the target voice
  • Enter the text to be synthesized
  • Kick off the cloning job and watch real‑time logs / progress
  • Play the generated WAV when done

Back‑end stack
  * Whisper‑cpp / OpenAI‑Whisper (transcription, optional)
  * Resemblyzer (speaker embedding)
  * Coqui‑TTS XTTS‑v2 (multilingual zero‑shot TTS)

The heavy lifting runs in a QThread so the UI stays responsive.

Requires:
  pip install "PySide6>=6.6" pydub numpy resemblyzer TTS whisper ffmpeg-python
"""

import sys
import logging
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from pydub import AudioSegment
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QProgressBar, QMessageBox
)

from resemblyzer import preprocess_wav, VoiceEncoder
from TTS.api import TTS
import whisper

VOICE_DIR = Path("voices"); VOICE_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------  BACK‑END  ------------------------------------ #

def convert_to_wav(src: Path) -> Path:
    """Convert MP3/other to 16‑kHz mono WAV suitable for embedding."""
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".wav")
    audio = AudioSegment.from_file(src)
    audio.set_channels(1).set_frame_rate(16000).export(dst, format="wav")
    return dst


def get_embedding(wav: Path) -> np.ndarray:
    enc = VoiceEncoder()
    wav_arr = preprocess_wav(wav)
    return enc.embed_utterance(wav_arr)


def synthesize(text: str, speaker_wavs: List[Path], outfile: Path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=[str(p) for p in speaker_wavs],
        file_path=str(outfile)
    )

# ---------------------------  WORKER  -------------------------------------- #
class CloneWorker(QThread):
    log = Signal(str)
    done = Signal(Path)
    progress = Signal(int)

    def __init__(self, refs: List[str], text: str):
        super().__init__()
        self.refs = [Path(p) for p in refs]
        self.text = text

    def run(self):
        try:
            self.log.emit("Starting voice‑clone job …")
            wavs = []
            for idx, f in enumerate(self.refs, 1):
                self.log.emit(f"[Stage 1] Converting {f.name} → WAV")
                wav = convert_to_wav(f)
                wavs.append(wav)
                self.progress.emit(int(idx / len(self.refs) * 30))

            # optional transcription (diagnostic)
            self.log.emit("[Stage 2] Transcribing first clip with Whisper (optional)…")
            whisper_model = whisper.load_model("base")
            txt = whisper_model.transcribe(str(wavs[0]))["text"].strip()
            self.log.emit(f"Whisper says: '{txt}'")
            self.progress.emit(40)

            self.log.emit("[Stage 3] Generating speaker embeddings (for cache)…")
            for w in wavs:
                emb = get_embedding(w)
                np.save(MODEL_DIR / f"{w.stem}.npy", emb)
            self.progress.emit(60)

            out_path = OUTPUT_DIR / f"{wavs[0].stem}_clone.wav"
            self.log.emit("[Stage 4] Synthesizing speech with XTTS‑v2 … this may take a bit")
            synthesize(self.text, wavs, out_path)
            self.progress.emit(100)
            self.log.emit("✔ Voice synthesis complete.")
            self.done.emit(out_path)
        except Exception as e:
            self.log.emit(f"❌ ERROR: {e}")

# ---------------------------  GUI  ----------------------------------------- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zero‑Shot Voice Cloner")
        self.resize(600, 400)

        self.layout = QVBoxLayout(self)

        # File selection
        file_row = QHBoxLayout()
        self.file_label = QLabel("No reference selected")
        choose_btn = QPushButton("Choose reference clips…")
        choose_btn.clicked.connect(self.choose_files)
        file_row.addWidget(choose_btn)
        file_row.addWidget(self.file_label)
        self.layout.addLayout(file_row)

        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type the text to be spoken here …")
        self.layout.addWidget(self.text_input)

        # Progress + logs
        self.progress = QProgressBar(); self.layout.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)

        # Action buttons
        run_btn = QPushButton("Clone Voice →")
        run_btn.clicked.connect(self.run_clone)
        self.layout.addWidget(run_btn)

        self.refs: List[str] = []
        self.worker: CloneWorker | None = None

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select reference audio", str(VOICE_DIR),
                                                "Audio Files (*.mp3 *.wav *.m4a *.flac)")
        if paths:
            self.refs = paths
            self.file_label.setText(f"{len(paths)} file(s) selected")

    def run_clone(self):
        if not self.refs:
            QMessageBox.warning(self, "Missing reference", "Please select at least one reference clip.")
            return
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing text", "Please enter the text to synthesize.")
            return

        self.progress.setValue(0); self.log_box.clear()
        self.worker = CloneWorker(self.refs, text)
        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self.play_output)
        self.worker.start()

    def play_output(self, wav: Path):
        self.log_box.append(f"Playing {wav} …")
        # cross‑platform playback via ffplay (ffmpeg) if present
        try:
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", str(wav)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            QMessageBox.information(self, "Done", f"Output saved to {wav}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    mw = MainWindow(); mw.show()
    sys.exit(app.exec())
