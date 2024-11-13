import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
from text.symbols import symbols, num_languages, num_tones
from sentence_transformers import SentenceTransformer
import numpy as np

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-pct", default=0.2)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    val_pct: float,
    config_path: str,
    max_val_total: int,
    clean: bool,
):
    
    e5_model = SentenceTransformer('intfloat/multilingual-e5-base').cuda()

    all_speaker_descrips = []
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                #print(line)
                utt, spk, language, text = line.strip().split("|")
    
                norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda:0')
                for ph in phones:
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print('update!, now symbols:')
                        print(new_symbols)
                        with open(f'{language}_symbol.txt', 'w') as f:
                            f.write(f'{new_symbols}')
    
                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                bert_path = utt.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)
                all_speaker_descrips.append(spk)
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        metadata = cleaned_path
        
    # first embed all of the speaker descriptions
    speaker_embeddings = e5_model.encode(all_speaker_descrips, normalize_embeddings=True, batch_size=128)
        
    all_lines = []
    with open(metadata, encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk = str(np.squeeze(speaker_embeddings[i]).tolist())
            all_lines.append(f"{utt}|{spk}|{language}|{text}|{phones}|{tones}|{word2ph}\n")

    # just split train and val based on val_pct
    shuffle(all_lines)
    split = int(val_pct * len(all_lines))
    train_list = all_lines[split:]
    val_list = all_lines[:split]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    # config["data"]["spk2id"] = spk_id_map

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = 0
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols
    config["model"]["use_spk_conditioned_encoder"] = False

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
