  GNU nano 8.1                                                                                                   voice_sampler.py                                                                                                             
#!/usr/bin/env python3
import argparse
from pathlib import Path

from lib.convert import convert_mp3_to_wav
from lib.embed import create_embedding
from lib.synthesize import synthesize

def main():
    parser = argparse.ArgumentParser(description="Voice Sampler + Synthesizer")
    parser.add_argument("--voice", required=True, help="Path to input MP3")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Path to output WAV")
    args = parser.parse_args()

    voice_path = Path(args.voice).resolve()
    wavs_dir = Path("wavs")
    embeddings_dir = Path("embeddings")
    output_path = Path(args.output).resolve()

    # Ensure necessary directories exist
    wavs_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert MP3 to WAV
    wav_path = convert_mp3_to_wav(voice_path, wavs_dir)

    # Generate speaker embedding
    embedding_path = embeddings_dir / f"{voice_path.stem}.npy"
    create_embedding(wav_path, embedding_path)

    # Synthesize output
    synthesize(args.text, embedding_path, output_path)
    print(f"✅ Synthesized speech saved to: {output_path}")

if __name__ == "__main__":
    main()


