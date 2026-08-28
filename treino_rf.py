from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


B2W_CSV = Path("data/raw/b2w_reviews_sample.csv")
TG_CSV = Path("data/output/respostas_ias_tg.csv")
OUTPUT_CSV = Path(
    "data/processed/resultados_random_forest.csv"
)

B2W_URL = (
    "https://raw.githubusercontent.com/"
    "b2wdigital/b2w-reviews01/master/"
    "B2W-Reviews01.csv"
)

RANDOM_STATE = 42


def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()

    texto = re.sub(
        r"[^a-záàâãéèêíóòôõúç\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def categorizar_rating(rating):
    if rating <= 2:
        return "NEGATIVO"

    if rating == 3:
        return "NEUTRO"

    return "POSITIVO"


def carregar_b2w():
    if B2W_CSV.exists():
        return pd.read_csv(
            B2W_CSV,
            encoding="utf-8-sig"
        )

    print(
        "Baixando amostra do B2W-Reviews01..."
    )

    df_raw = pd.read_csv(
        B2W_URL,
        low_memory=False
    ).dropna(
        subset=[
            "review_text",
            "overall_rating"
        ]
    )

    df_raw = df_raw.sample(
        n=min(10000, len(df_raw)),
        random_state=RANDOM_STATE
    )

    B2W_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_raw[
        [
            "review_text",
            "overall_rating"
        ]
    ].to_csv(
        B2W_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    return df_raw[
        [
            "review_text",
            "overall_rating"
        ]
    ].copy()


def main():
    print("=" * 65)
    print("RANDOM FOREST + TF-IDF")
    print("=" * 65)

    nltk.download(
        "stopwords",
        quiet=True
    )

    # Preserva negações importantes para sentimento.
    negacoes = {
        "não",
        "nem",
        "nunca",
        "jamais",
    }

    stop_words_pt = [
        palavra
        for palavra
        in stopwords.words("portuguese")
        if palavra not in negacoes
    ]

    df_b2w = carregar_b2w()

    df_b2w["label"] = (
        df_b2w["overall_rating"]
        .apply(categorizar_rating)
    )

    df_b2w["text_clean"] = (
        df_b2w["review_text"]
        .apply(limpar_texto)
    )

    vectorizer = TfidfVectorizer(
        stop_words=stop_words_pt,
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df_b2w["text_clean"]
    )

    y = df_b2w["label"]

    (
        X_treino,
        X_teste,
        y_treino,
        y_teste
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_model.fit(
        X_treino,
        y_treino
    )

    y_pred = rf_model.predict(
        X_teste
    )

    print()
    print(
        f"Acurácia no conjunto de teste: "
        f"{accuracy_score(y_teste, y_pred) * 100:.2f}%"
    )

    print()
    print(
        classification_report(
            y_teste,
            y_pred
        )
    )

    if not TG_CSV.exists():
        raise FileNotFoundError(
            f"Respostas do TG não encontradas: {TG_CSV.resolve()}"
        )

    df_tg = pd.read_csv(
        TG_CSV,
        encoding="utf-8-sig"
    )

    colunas_necessarias = {
        "response_id",
        "resposta",
        "status"
    }

    faltando = (
        colunas_necessarias
        - set(df_tg.columns)
    )

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltando)}"
        )

    df_tg = df_tg[
        df_tg["status"]
        .astype(str)
        .str.upper()
        == "OK"
    ].copy()

    df_tg["resposta_limpa"] = (
        df_tg["resposta"]
        .fillna("")
        .apply(limpar_texto)
    )

    df_tg = df_tg[
        df_tg["resposta_limpa"] != ""
    ].copy()

    X_tg = vectorizer.transform(
        df_tg["resposta_limpa"]
    )

    predicoes = rf_model.predict(
        X_tg
    )

    probabilidades = (
        rf_model
        .predict_proba(X_tg)
        .max(axis=1)
    )

    resultados = pd.DataFrame(
        {
            "response_id":
                df_tg["response_id"].values,

            "rf_label":
                predicoes,

            "rf_score":
                probabilidades.round(6),
        }
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 65)
    print("CLASSIFICAÇÃO DAS RESPOSTAS CONCLUÍDA")
    print("=" * 65)
    print(
        f"Respostas classificadas: "
        f"{len(resultados)}"
    )
    print()
    print(
        resultados["rf_label"]
        .value_counts()
    )
    print()
    print(
        f"Arquivo salvo em: "
        f"{OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
