import os
import click
from melo.api import TTS
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

"""
python infer.py \
    -m data/tts/melo-data/logs/melo-data/G_20000.pth \
    -t "Hello, my name is Melo" \
    -s "A female voice, with a very high pitch, speaks in a monotone manner. The recording quality is very clear and close-sounding, indicating a good or excellent audio capture." \
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
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    e5_model = SentenceTransformer('intfloat/multilingual-e5-base').cuda()

    encoded_descrip = np.squeeze(e5_model.encode([speaker_description], normalize_embeddings=True)[0])
    
    save_path = f'{output_dir}/output.wav'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.tts_to_file(text, 0, save_path, e5_embeddings=encoded_descrip)

if __name__ == "__main__":
    main()
