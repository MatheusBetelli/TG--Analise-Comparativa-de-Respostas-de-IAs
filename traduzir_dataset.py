from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


INPUT_CSV = Path("data/input/mensagens_tg_selecionadas.csv")
OUTPUT_CSV = Path("data/input/mensagens_tg.csv")

# Modelo independente dos três LLMs avaliados.
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"
TARGET_PREFIX = ">>pt_BR<< "


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo de seleção não encontrado: {INPUT_CSV.resolve()}"
        )

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    colunas_necessarias = {
        "mensagem_id",
        "numero_mensagem_original",
        "categoria",
        "mensagem_original",
    }

    faltando = colunas_necessarias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltando)}"
        )

    traducoes_existentes = {}

    # Permite retomar caso o processo seja interrompido.
    if OUTPUT_CSV.exists():
        antigo = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")

        if {"mensagem_id", "mensagem_pt"}.issubset(antigo.columns):
            for _, row in antigo.iterrows():
                valor = row.get("mensagem_pt")

                if pd.notna(valor) and str(valor).strip():
                    traducoes_existentes[
                        str(row["mensagem_id"])
                    ] = str(valor).strip()

    print("Carregando modelo de tradução...")
    print(f"Modelo: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Dispositivo: {device}")
    print()

    traducoes = []

    for indice, row in df.iterrows():
        mensagem_id = str(row["mensagem_id"])
        original = str(row["mensagem_original"]).strip()

        if mensagem_id in traducoes_existentes:
            traducao = traducoes_existentes[mensagem_id]

            print(
                f"[{indice + 1}/{len(df)}] "
                f"{mensagem_id} -> já traduzida"
            )

        else:
            entrada = TARGET_PREFIX + original

            encoded = tokenizer(
                entrada,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            encoded = {
                chave: valor.to(device)
                for chave, valor in encoded.items()
            }

            with torch.no_grad():
                gerado = model.generate(
                    **encoded,
                    max_new_tokens=512,
                    num_beams=4
                )

            traducao = tokenizer.decode(
                gerado[0],
                skip_special_tokens=True
            ).strip()

            print(
                f"[{indice + 1}/{len(df)}] "
                f"{mensagem_id} -> OK"
            )

        traducoes.append(traducao)

        # Salva progresso após cada tradução.
        parcial = df.copy()
        parcial["mensagem_pt"] = (
            traducoes
            + [""] * (len(df) - len(traducoes))
        )

        OUTPUT_CSV.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        parcial.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    print()
    print("=" * 60)
    print("TRADUÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Mensagens traduzidas: {len(df)}")
    print(f"Arquivo salvo em: {OUTPUT_CSV.resolve()}")
    print()
    print(
        "Revise uma amostra das traduções antes da coleta oficial, "
        "principalmente para verificar se o tom emocional foi preservado."
    )


if __name__ == "__main__":
    main()
