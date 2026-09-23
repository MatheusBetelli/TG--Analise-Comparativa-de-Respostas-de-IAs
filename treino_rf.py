from pathlib import Path
import json
import re
import urllib.request

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


COMMENTS_PATH = Path("data/raw/comments.json")
SENTILEX_PATH = Path("data/raw/SentiLex-flex-PT02.txt")
TG_CSV = Path("data/output/respostas_ias_tg.csv")
OUTPUT_CSV = Path("data/processed/resultados_random_forest.csv")

COMMENTS_URL = (
    "https://raw.githubusercontent.com/"
    "MaiconChavesMarques/BrStudentMH-dataset/main/"
    "comments.json"
)

SENTILEX_URL = (
    "https://raw.githubusercontent.com/"
    "sillasgonzaga/lexiconPT/master/data-raw/"
    "SentiLex-flex-PT02.txt"
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


def baixar_se_necessario(url: str, destino: Path):
    if destino.exists():
        return

    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {destino.name} de {url}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    print(f"{destino.name} baixado com sucesso.")


def carregar_sentilex():
    baixar_se_necessario(SENTILEX_URL, SENTILEX_PATH)

    print("Carregando léxico de sentimento SentiLex-PT...")
    lexicon = {}

    with open(SENTILEX_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split(".")
            if len(parts) >= 2:
                word = parts[0].strip().lower().split(",")[0]
                m = re.search(r"POL:N0=([-0-9]+)", line)
                if m:
                    lexicon[word] = int(m.group(1))

    print(f"SentiLex carregado com {len(lexicon)} palavras anotadas.")
    return lexicon


def carregar_dados_estudantes(lexicon):
    baixar_se_necessario(COMMENTS_URL, COMMENTS_PATH)

    print("Carregando comentários de estudantes do BrStudentMH...")
    with open(COMMENTS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        comments_data = json.loads(f.read(), strict=False)

    registros = []
    for c in comments_data:
        body = c.get("body", "")
        texto_limpo = limpar_texto(body)
        palavras = texto_limpo.split()

        # Filtra comentários curtos demais para conter sinal de sentimento
        if len(palavras) < 5:
            continue

        score_sentimento = sum(lexicon.get(w, 0) for w in palavras)

        if score_sentimento > 0:
            rotulo = "POSITIVO"
        elif score_sentimento < 0:
            rotulo = "NEGATIVO"
        else:
            rotulo = "NEUTRO"

        registros.append(
            {
                "text_clean": texto_limpo,
                "label": rotulo,
            }
        )

    df = pd.DataFrame(registros)
    print(f"\nTotal de comentários de estudantes rotulados: {len(df)}")
    print("Distribuição inicial dos sentimentos:")
    print(df["label"].value_counts())

    # Balanceamento das classes para evitar viés no Random Forest
    min_amostras = min(3500, df["label"].value_counts().min())
    dfs = [
        df[df["label"] == rotulo].sample(
            n=min_amostras,
            random_state=RANDOM_STATE
        )
        for rotulo in ["POSITIVO", "NEUTRO", "NEGATIVO"]
    ]
    df_balanceado = pd.concat(dfs, ignore_index=True)

    print(f"\nBase balanceada para treino: {len(df_balanceado)} comentários ({min_amostras} por classe).")
    return df_balanceado


def main():
    print("=" * 65)
    print("RANDOM FOREST + TF-IDF (ANÁLISE DE SENTIMENTO)")
    print("Treinado com comentários de estudantes (BrStudentMH + SentiLex-PT)")
    print("=" * 65)

    nltk.download(
        "stopwords",
        quiet=True
    )

    # Preserva negações cruciais para a análise de sentimento.
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

    lexicon = carregar_sentilex()
    df_treino = carregar_dados_estudantes(lexicon)

    vectorizer = TfidfVectorizer(
        stop_words=stop_words_pt,
        max_features=4000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df_treino["text_clean"]
    )

    y = df_treino["label"]

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

    print("\nTreinando Random ForestClassifier...")
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
            f"Colunas ausentes em {TG_CSV.name}: {sorted(faltando)}"
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
    print("CLASSIFICAÇÃO DE SENTIMENTO DAS RESPOSTAS CONCLUÍDA")
    print("=" * 65)
    print(
        f"Respostas classificadas: "
        f"{len(resultados)}"
    )
    print()
    print("Distribuição dos sentimentos preditos (Random Forest):")
    print(
        resultados["rf_label"]
        .value_counts()
    )
    print()
    print(
        f"Arquivo salvo com sucesso em: "
        f"{OUTPUT_CSV.resolve()}"
    )

    try:
        from gerar_graficos import gerar_graficos
        print()
        gerar_graficos()
    except Exception as e:
        print(f"\nAviso: Não foi possível gerar gráficos automáticos: {e}")


if __name__ == "__main__":
    main()
