from pathlib import Path

import pandas as pd
from transformers import pipeline


# ============================================================
# ARQUIVOS
# ============================================================

INPUT_CSV = Path(
    "data/output/respostas_ias_tg.csv"
)

OUTPUT_CSV = Path(
    "data/processed/resultados_caramelo.csv"
)


# ============================================================
# MODELO
# ============================================================

MODEL_NAME = "Adilmar/caramelo-smile"


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("Carregando Caramelo-Smile...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME
    )

    print("Modelo carregado.")


    # ========================================================
    # CARREGA AS RESPOSTAS
    # ========================================================

    df = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig"
    )

    print(
        f"Respostas encontradas: {len(df)}"
    )


    # ========================================================
    # LISTA DOS RESULTADOS
    # ========================================================

    resultados_finais = []


    # ========================================================
    # CLASSIFICA CADA RESPOSTA
    # ========================================================

    for indice, row in df.iterrows():

        texto = str(
            row["resposta"]
        )

        resultado = classifier(
            texto,
            truncation=True,
            max_length=512
        )[0]


        classificacao = resultado[
            "label"
        ]

        score = resultado[
            "score"
        ]


        # ====================================================
        # GUARDA SOMENTE O QUE INTERESSA
        # ====================================================

        resultados_finais.append({

            "response_id":
                row["response_id"],

            "classificacao":
                classificacao,

            "score":
                round(score, 6)

        })


        print(
            f"[{indice + 1}/{len(df)}] "
            f"{row['response_id']} -> "
            f"{classificacao} "
            f"({score:.4f})"
        )


    # ========================================================
    # CRIA DATAFRAME SOMENTE COM RESULTADOS
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados_finais
    )


    # ========================================================
    # CRIA PASTA
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # SALVA CSV
    # ========================================================

    df_resultados.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # RESUMO
    # ========================================================

    print()

    print("=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)

    print(
        f"Respostas analisadas: "
        f"{len(df_resultados)}"
    )

    print()

    print(
        "Distribuição das classificações:"
    )

    print(
        df_resultados[
            "classificacao"
        ].value_counts()
    )

    print()

    print(
        f"Arquivo salvo em:"
    )

    print(
        OUTPUT_CSV.resolve()
    )


if __name__ == "__main__":
    main()