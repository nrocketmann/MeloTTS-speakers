import os
import click
from melo.api import TTS
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

"""
python infer.py \
    -m /home/jovyan/data/tts/melo-data/logs/melo-data/G_55000.pth \
    -t "At the dawn of time there was but one being, not born but rather there for all eternity, ruling over a kingdom of vast emptiness." \
    -s "A male speaker delivers expressive, high-pitched lines in a far-away sounding recording" \
    -l "EN" \
    -o "outputs"

"""
    
    
@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--speaker-description', '-s', type=str, default=None, help="Speaker description")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
def main(ckpt_path, text, speaker_description, language, output_dir):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")
    
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    print(config_path)
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    e5_model = SentenceTransformer('intfloat/multilingual-e5-base').cuda()

    speaker_descrips = [
        "A female voice, with a very high pitch, speaks in a monotone manner. The recording quality is very clear and close-sounding, indicating a good or excellent audio capture.",
        "A female speaker delivers a slightly expressive and animated tone with a clear and closely recorded voice. The moderate pace of her speech enhances the overall clarity.",
        "A female speaker",
        "A male speaker",
        "A female speaker with a high pitched voice",
        "A female speaker with a low pitched voice",
        "A male speaker with a high pitched voice",
        "A highly expressive very angry male speaker shouting loudly",
        "A shy female speaker whispering",
    ]

    # encoded_descrip = np.squeeze(e5_model.encode([speaker_description], normalize_embeddings=True)[0])

    encoded_descrips = e5_model.encode(speaker_descrips, normalize_embeddings=True)

    for i, descrip in enumerate(encoded_descrips):
        save_path = f'{output_dir}/output_{i}.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, 5, save_path, e5_embeddings=np.squeeze(descrip))

if __name__ == "__main__":
    main()
