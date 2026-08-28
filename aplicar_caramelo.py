from pathlib import Path

import pandas as pd
from transformers import pipeline


INPUT_CSV = Path(
    "data/output/respostas_ias_tg.csv"
)

OUTPUT_CSV = Path(
    "data/processed/resultados_caramelo.csv"
)

MODEL_NAME = "Adilmar/caramelo-smile"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo de respostas não encontrado: {INPUT_CSV.resolve()}"
        )

    print("=" * 60)
    print("ANÁLISE COM CARAMELO-SMILE")
    print("=" * 60)

    print("\nCarregando Caramelo-Smile...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME
    )

    print("Modelo carregado.")

    df = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig"
    )

    colunas_necessarias = {
        "response_id",
        "resposta",
        "status",
    }

    faltando = (
        colunas_necessarias
        - set(df.columns)
    )

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltando)}"
        )

    # Analisa somente respostas coletadas com sucesso.
    df = df[
        df["status"].astype(str).str.upper()
        == "OK"
    ].copy()

    df["resposta"] = (
        df["resposta"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["resposta"] != ""
    ].copy()

    print(
        f"Respostas válidas encontradas: {len(df)}"
    )

    resultados_finais = []

    for posicao, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):
        texto = row["resposta"]

        resultado = classifier(
            texto,
            truncation=True,
            max_length=512
        )[0]

        classificacao = resultado["label"]
        score = resultado["score"]

        resultados_finais.append(
            {
                "response_id":
                    row["response_id"],

                "classificacao":
                    classificacao,

                "score":
                    round(float(score), 6),
            }
        )

        print(
            f"[{posicao}/{len(df)}] "
            f"{row['response_id']} -> "
            f"{classificacao} "
            f"({score:.4f})"
        )

    df_resultados = pd.DataFrame(
        resultados_finais
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_resultados.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)
    print(
        f"Respostas analisadas: "
        f"{len(df_resultados)}"
    )
    print()
    print("Distribuição das classificações:")
    print(
        df_resultados[
            "classificacao"
        ].value_counts()
    )
    print()
    print(
        f"Arquivo salvo em: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
