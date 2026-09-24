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


# ============================================================
# CAMINHOS DOS ARQUIVOS
# ============================================================

COMMENTS_PATH = Path("data/raw/comments.json")
SENTILEX_PATH = Path("data/raw/SentiLex-flex-PT02.txt")

TG_CSV = Path("data/output/respostas_ias_tg.csv")

OUTPUT_CSV = Path(
    "data/processed/resultados_random_forest.csv"
)


# ============================================================
# URLs
# ============================================================

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


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42

MAX_AMOSTRAS_POR_CLASSE = 3500

TEST_SIZE = 0.20


# ============================================================
# LIMPEZA DE TEXTO
# ============================================================

def limpar_texto(texto):

    if not isinstance(texto, str):
        return ""

    texto = texto.lower()

    # Mantém letras e acentos
    texto = re.sub(
        r"[^a-záàâãéèêíóòôõúç\s]",
        " ",
        texto
    )

    # Remove espaços duplicados
    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


# ============================================================
# DOWNLOAD DE ARQUIVOS
# ============================================================

def baixar_se_necessario(url: str, destino: Path):

    if destino.exists():
        return

    destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Baixando {destino.name}..."
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req) as resp, \
            open(destino, "wb") as f:

        while True:

            chunk = resp.read(
                1024 * 1024
            )

            if not chunk:
                break

            f.write(chunk)

    print(
        f"{destino.name} baixado com sucesso."
    )


# ============================================================
# CARREGAR SENTILEX
# ============================================================

def carregar_sentilex():

    baixar_se_necessario(
        SENTILEX_URL,
        SENTILEX_PATH
    )

    print(
        "\nCarregando léxico de sentimento "
        "SentiLex-PT..."
    )

    lexicon = {}

    with open(
        SENTILEX_PATH,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            parts = line.split(".")

            if len(parts) >= 2:

                palavra = (
                    parts[0]
                    .strip()
                    .lower()
                    .split(",")[0]
                )

                match = re.search(
                    r"POL:N0=([-0-9]+)",
                    line
                )

                if match:

                    lexicon[palavra] = int(
                        match.group(1)
                    )

    print(
        f"SentiLex carregado com "
        f"{len(lexicon)} palavras anotadas."
    )

    return lexicon


# ============================================================
# CARREGAR E ROTULAR BRSTUDENTMH
# ============================================================

def carregar_dados_estudantes(lexicon):

    baixar_se_necessario(
        COMMENTS_URL,
        COMMENTS_PATH
    )

    print(
        "\nCarregando comentários "
        "do BrStudentMH..."
    )

    with open(
        COMMENTS_PATH,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        comments_data = json.loads(
            f.read(),
            strict=False
        )

    registros = []

    for comentario in comments_data:

        body = comentario.get(
            "body",
            ""
        )

        texto_limpo = limpar_texto(
            body
        )

        palavras = texto_limpo.split()

        # Ignora comentários muito curtos
        if len(palavras) < 5:
            continue

        # Soma a polaridade das palavras
        # encontradas no SentiLex
        score_sentimento = sum(
            lexicon.get(palavra, 0)
            for palavra in palavras
        )

        if score_sentimento > 0:

            rotulo = "POSITIVO"

        elif score_sentimento < 0:

            rotulo = "NEGATIVO"

        else:

            rotulo = "NEUTRO"

        registros.append(
            {
                "text_clean":
                    texto_limpo,

                "label":
                    rotulo
            }
        )

    df = pd.DataFrame(
        registros
    )

    # Remove possíveis textos duplicados
    df = df.drop_duplicates(
        subset=["text_clean"]
    ).reset_index(
        drop=True
    )

    print(
        f"\nTotal de comentários "
        f"rotulados: {len(df)}"
    )

    print(
        "\nDistribuição inicial "
        "dos sentimentos:"
    )

    print(
        df["label"]
        .value_counts()
    )


    # ========================================================
    # BALANCEAMENTO DAS CLASSES
    # ========================================================

    quantidade_menor_classe = (
        df["label"]
        .value_counts()
        .min()
    )

    min_amostras = min(
        MAX_AMOSTRAS_POR_CLASSE,
        quantidade_menor_classe
    )

    partes = []

    for rotulo in [
        "POSITIVO",
        "NEUTRO",
        "NEGATIVO"
    ]:

        amostra = (
            df[
                df["label"] == rotulo
            ]
            .sample(
                n=min_amostras,
                random_state=RANDOM_STATE
            )
        )

        partes.append(
            amostra
        )

    df_balanceado = pd.concat(
        partes,
        ignore_index=True
    )

    # Embaralha a base
    df_balanceado = (
        df_balanceado
        .sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nBase balanceada:"
    )

    print(
        f"{len(df_balanceado)} comentários"
    )

    print(
        f"{min_amostras} comentários "
        "por classe."
    )

    print(
        "\nDistribuição após "
        "balanceamento:"
    )

    print(
        df_balanceado[
            "label"
        ].value_counts()
    )

    return df_balanceado


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "RANDOM FOREST + TF-IDF"
    )

    print(
        "ANÁLISE DE SENTIMENTO"
    )

    print(
        "BrStudentMH + SentiLex-PT"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # STOPWORDS
    # ========================================================

    nltk.download(
        "stopwords",
        quiet=True
    )

    # Palavras de negação são importantes
    # para análise de sentimentos
    negacoes = {
        "não",
        "nem",
        "nunca",
        "jamais"
    }

    stop_words_pt = [
        palavra
        for palavra
        in stopwords.words(
            "portuguese"
        )
        if palavra not in negacoes
    ]


    # ========================================================
    # CARREGAR DADOS
    # ========================================================

    lexicon = carregar_sentilex()

    df_treino = (
        carregar_dados_estudantes(
            lexicon
        )
    )


    # ========================================================
    # SEPARAÇÃO TREINO / TESTE
    #
    # IMPORTANTE:
    # PRIMEIRO fazemos o train_test_split.
    # O TF-IDF ainda NÃO foi treinado aqui.
    # ========================================================

    X_texto = (
        df_treino[
            "text_clean"
        ]
    )

    y = (
        df_treino[
            "label"
        ]
    )

    (
        X_treino_texto,
        X_teste_texto,
        y_treino,
        y_teste
    ) = train_test_split(

        X_texto,
        y,

        test_size=TEST_SIZE,

        random_state=
            RANDOM_STATE,

        stratify=y
    )


    print(
        "\nDivisão dos dados:"
    )

    print(
        f"Treino: "
        f"{len(X_treino_texto)}"
    )

    print(
        f"Teste: "
        f"{len(X_teste_texto)}"
    )


    print(
        "\nDistribuição das classes "
        "no treino:"
    )

    print(
        y_treino
        .value_counts()
    )


    print(
        "\nDistribuição das classes "
        "no teste:"
    )

    print(
        y_teste
        .value_counts()
    )


    # ========================================================
    # TF-IDF
    #
    # O TF-IDF aprende SOMENTE com o conjunto de treino.
    # ========================================================

    print(
        "\nCriando representação TF-IDF..."
    )

    vectorizer = TfidfVectorizer(

        stop_words=
            stop_words_pt,

        max_features=4000,

        ngram_range=(1, 2),

        sublinear_tf=True
    )


    # FIT somente no TREINO
    X_treino = (
        vectorizer
        .fit_transform(
            X_treino_texto
        )
    )


    # TESTE apenas é transformado
    X_teste = (
        vectorizer
        .transform(
            X_teste_texto
        )
    )


    print(
        f"Quantidade de features "
        f"TF-IDF: "
        f"{len(vectorizer.get_feature_names_out())}"
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nTreinando "
        "Random ForestClassifier..."
    )

    rf_model = (
        RandomForestClassifier(

            n_estimators=150,

            random_state=
                RANDOM_STATE,

            class_weight=
                "balanced",

            n_jobs=-1
        )
    )


    rf_model.fit(
        X_treino,
        y_treino
    )


    # ========================================================
    # AVALIAÇÃO NO CONJUNTO DE TESTE
    # ========================================================

    y_pred = rf_model.predict(
        X_teste
    )


    acuracia = accuracy_score(
        y_teste,
        y_pred
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTADOS NO CONJUNTO DE TESTE"
    )

    print(
        "=" * 70
    )


    print(
        f"\nAcurácia: "
        f"{acuracia * 100:.2f}%"
    )


    print(
        "\nRelatório de classificação:"
    )

    print(
        classification_report(

            y_teste,
            y_pred,

            digits=4,

            zero_division=0
        )
    )


    # ========================================================
    # CARREGAR AS 300 RESPOSTAS DO TG
    # ========================================================

    if not TG_CSV.exists():

        raise FileNotFoundError(
            "Arquivo com as respostas "
            f"não encontrado: "
            f"{TG_CSV.resolve()}"
        )


    print(
        "\nCarregando respostas "
        "das Inteligências Artificiais..."
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

            "Colunas ausentes no CSV: "
            f"{sorted(faltando)}"
        )


    # Utiliza apenas respostas
    # coletadas com sucesso
    df_tg = (

        df_tg[

            df_tg[
                "status"
            ]
            .astype(str)
            .str.upper()

            == "OK"
        ]
        .copy()
    )


    # ========================================================
    # LIMPEZA DAS RESPOSTAS
    # ========================================================

    df_tg[
        "resposta_limpa"
    ] = (

        df_tg[
            "resposta"
        ]
        .fillna("")
        .apply(
            limpar_texto
        )
    )


    df_tg = (

        df_tg[
            df_tg[
                "resposta_limpa"
            ]
            != ""
        ]
        .copy()
    )


    # ========================================================
    # TRANSFORMAR AS RESPOSTAS
    #
    # ATENÇÃO:
    # usamos TRANSFORM, não FIT_TRANSFORM.
    #
    # O TF-IDF continua sendo exatamente
    # aquele aprendido no conjunto de treino.
    # ========================================================

    X_tg = (
        vectorizer
        .transform(

            df_tg[
                "resposta_limpa"
            ]
        )
    )


    # ========================================================
    # CLASSIFICAÇÃO DAS RESPOSTAS
    # ========================================================

    predicoes = (
        rf_model
        .predict(
            X_tg
        )
    )


    probabilidades = (

        rf_model
        .predict_proba(
            X_tg
        )

        .max(
            axis=1
        )
    )


    resultados = pd.DataFrame(
        {

            "response_id":
                df_tg[
                    "response_id"
                ].values,

            "rf_label":
                predicoes,

            "rf_score":
                probabilidades.round(6)
        }
    )


    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    OUTPUT_CSV.parent.mkdir(

        parents=True,

        exist_ok=True
    )


    resultados.to_csv(

        OUTPUT_CSV,

        index=False,

        encoding="utf-8-sig"
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLASSIFICAÇÃO DAS "
        "RESPOSTAS CONCLUÍDA"
    )

    print(
        "=" * 70
    )


    print(
        f"\nRespostas classificadas: "
        f"{len(resultados)}"
    )


    print(
        "\nDistribuição dos sentimentos:"
    )


    print(
        resultados[
            "rf_label"
        ]
        .value_counts()
    )


    print(
        "\nDistribuição percentual:"
    )


    print(

        (
            resultados[
                "rf_label"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


    print(
        "\nArquivo salvo em:"
    )

    print(
        OUTPUT_CSV.resolve()
    )


    # ========================================================
    # GERAR GRÁFICOS, SE O SCRIPT EXISTIR
    # ========================================================

    try:

        from gerar_graficos import gerar_graficos

        print(
            "\nGerando gráficos..."
        )

        gerar_graficos()

    except Exception as e:

        print(
            "\nAviso: "
            "não foi possível gerar "
            f"os gráficos automaticamente: {e}"
        )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()