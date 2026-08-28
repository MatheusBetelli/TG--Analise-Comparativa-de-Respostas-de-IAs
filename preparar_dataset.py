from pathlib import Path

import pandas as pd


SOURCE_CSV = Path("data/raw/mental_health_padronizado.csv")
OUTPUT_CSV = Path("data/input/mensagens_tg_selecionadas.csv")

CATEGORIAS = ["Depression", "Loneliness", "Anxiety", "Stress"]
N_POR_CATEGORIA = 25
MIN_PALAVRAS = 20
MAX_PALAVRAS = 200
RANDOM_STATE = 42


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Dataset original não encontrado: {SOURCE_CSV.resolve()}"
        )

    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")

    colunas_necessarias = {"numero_mensagem", "categoria", "mensagem"}
    faltando = colunas_necessarias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes no dataset: {sorted(faltando)}"
        )

    print(f"Linhas originais: {len(df)}")

    df = df.dropna(subset=["categoria", "mensagem"]).copy()
    df["mensagem"] = df["mensagem"].astype(str).str.strip()
    df = df[df["mensagem"] != ""]

    # Remove mensagens duplicadas globalmente.
    df = df.drop_duplicates(subset=["mensagem"]).copy()

    # Mantém somente as quatro categorias definidas para o TG.
    df = df[df["categoria"].isin(CATEGORIAS)].copy()

    # Controla o tamanho das mensagens.
    df["quantidade_palavras"] = df["mensagem"].str.split().str.len()
    df = df[
        df["quantidade_palavras"].between(MIN_PALAVRAS, MAX_PALAVRAS)
    ].copy()

    partes = []

    for categoria in CATEGORIAS:
        grupo = df[df["categoria"] == categoria]

        if len(grupo) < N_POR_CATEGORIA:
            raise ValueError(
                f"A categoria {categoria} possui somente {len(grupo)} "
                f"mensagens após os filtros."
            )

        amostra = grupo.sample(
            n=N_POR_CATEGORIA,
            random_state=RANDOM_STATE
        ).copy()

        partes.append(amostra)

    final = pd.concat(partes, ignore_index=True)

    final.insert(
        0,
        "mensagem_id",
        [f"M{i:03d}" for i in range(1, len(final) + 1)]
    )

    final = final.rename(
        columns={
            "numero_mensagem": "numero_mensagem_original",
            "mensagem": "mensagem_original",
        }
    )

    final = final[
        [
            "mensagem_id",
            "numero_mensagem_original",
            "categoria",
            "mensagem_original",
        ]
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    final.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("SELEÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Mensagens selecionadas: {len(final)}")
    print()
    print(final["categoria"].value_counts())
    print()
    print(f"Arquivo salvo em: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
