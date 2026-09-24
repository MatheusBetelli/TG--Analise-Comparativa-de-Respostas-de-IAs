from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

TOTAL_RESPOSTAS_ESPERADO = 300

NOME_RF = "Random Forest + TF-IDF"
NOME_CARAMELO = "Caramelo-Smile-2"

DIR_OUTPUT = Path("data/processed/graficos")
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

RF_CSV = Path("data/processed/resultados_random_forest.csv")
CARAMELO_CSV = Path("data/processed/resultados_caramelo.csv")
TG_CSV = Path("data/output/respostas_ias_tg.csv")
HUMAN_CSV = Path("data/output/respostas_formulario_humanos.csv")
CHAVE_JSON = Path("data/processed/chave_cegamento.json")

COMPARACAO_CSV = Path(
    "data/processed/comparacao_rf_caramelo_smile_2.csv"
)


# ============================================================
# CONFIGURAÇÕES VISUAIS
# ============================================================

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
else:
    plt.style.use("default")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.size"] = 10


CORES_PROVEDORES = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d97706",
    "Meta": "#0284c7",
}

CORES_SENTIMENTO = {
    "Positivo": "#2ecc71",
    "Neutro": "#95a5a6",
    "Negativo": "#e74c3c",
}

ORDEM_PROVEDORES = [
    "OpenAI",
    "Anthropic",
    "Meta",
]

ORDEM_SENTIMENTOS = [
    "Positivo",
    "Neutro",
    "Negativo",
]

ORDEM_CATEGORIAS = [
    "Depressão",
    "Solidão",
    "Ansiedade",
    "Estresse",
]


MAPA_SENTIMENTO = {
    "POSITIVO": "Positivo",
    "NEUTRO": "Neutro",
    "NEGATIVO": "Negativo",
    "positive": "Positivo",
    "neutral": "Neutro",
    "negative": "Negativo",
}

MAPA_CATEGORIA = {
    "Depression": "Depressão",
    "Loneliness": "Solidão",
    "Anxiety": "Ansiedade",
    "Stress": "Estresse",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def verificar_arquivo(caminho):
    """
    Verifica se um arquivo necessário existe.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )


def verificar_colunas(df, colunas, nome_arquivo):
    """
    Confere se todas as colunas esperadas existem.
    """
    faltando = [
        coluna
        for coluna in colunas
        if coluna not in df.columns
    ]

    if faltando:
        raise ValueError(
            f"{nome_arquivo} não possui as colunas: {faltando}"
        )


def verificar_ids_unicos(df, nome):
    """
    Impede que response_id duplicado provoque junções incorretas.
    """
    if df["response_id"].duplicated().any():
        duplicados = (
            df.loc[
                df["response_id"].duplicated(),
                "response_id"
            ]
            .tolist()
        )

        raise ValueError(
            f"Há response_id duplicado em {nome}: "
            f"{duplicados[:10]}"
        )


def salvar_figura(fig, caminho):
    """
    Salva a figura em alta resolução.
    """
    fig.tight_layout()

    fig.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Salvo: {caminho}")


def anotar_barras_agrupadas(
    ax,
    tabela_qtd,
    tabela_pct,
    fontsize=8.5
):
    """
    Insere quantidade e percentual sobre barras agrupadas.
    """

    for coluna_idx, container in enumerate(ax.containers):

        labels = []

        for linha_idx, _ in enumerate(container):

            qtd = int(
                tabela_qtd.iloc[
                    linha_idx,
                    coluna_idx
                ]
            )

            pct = float(
                tabela_pct.iloc[
                    linha_idx,
                    coluna_idx
                ]
            )

            labels.append(
                f"{qtd}\n({pct:.1f}%)"
            )

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=fontsize,
            fontweight="bold",
            color="#2c3e50"
        )


def converter_likert(valor):
    """
    Converte respostas do Google Forms para valores 1-5.
    """

    mapa = {
        "1 — Discordo totalmente": 1,
        "2 — Discordo": 2,
        "3 — Neutro / parcialmente": 3,
        "4 — Concordo": 4,
        "5 — Concordo totalmente": 5,
    }

    if pd.isna(valor):
        return None

    if isinstance(valor, (int, float)):
        if 1 <= valor <= 5:
            return float(valor)

    return mapa.get(str(valor).strip())


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def gerar_graficos():

    print("=" * 70)
    print("GERAÇÃO DOS GRÁFICOS DO TRABALHO DE GRADUAÇÃO")
    print(
        "Random Forest + TF-IDF | "
        "Caramelo-Smile-2 | "
        "Avaliação Humana"
    )
    print("=" * 70)


    # ========================================================
    # VERIFICA ARQUIVOS
    # ========================================================

    arquivos = [
        RF_CSV,
        CARAMELO_CSV,
        TG_CSV,
        HUMAN_CSV,
        CHAVE_JSON,
    ]

    for arquivo in arquivos:
        verificar_arquivo(arquivo)


    # ========================================================
    # CARREGA DADOS
    # ========================================================

    df_rf = pd.read_csv(
        RF_CSV,
        encoding="utf-8-sig"
    )

    df_caramelo = pd.read_csv(
        CARAMELO_CSV,
        encoding="utf-8-sig"
    )

    df_tg = pd.read_csv(
        TG_CSV,
        encoding="utf-8-sig"
    )


    # ========================================================
    # VALIDAÇÃO DAS COLUNAS
    # ========================================================

    verificar_colunas(
        df_rf,
        [
            "response_id",
            "rf_label",
        ],
        "resultados_random_forest.csv"
    )

    verificar_colunas(
        df_caramelo,
        [
            "response_id",
            "classificacao",
            "score",
        ],
        "resultados_caramelo.csv"
    )

    verificar_colunas(
        df_tg,
        [
            "response_id",
            "categoria",
            "provedor",
        ],
        "respostas_ias_tg.csv"
    )


    # ========================================================
    # VALIDAÇÃO DOS IDs
    # ========================================================

    verificar_ids_unicos(
        df_rf,
        "Random Forest"
    )

    verificar_ids_unicos(
        df_caramelo,
        "Caramelo-Smile-2"
    )

    verificar_ids_unicos(
        df_tg,
        "respostas_ias_tg.csv"
    )


    # ========================================================
    # CONFERE QUANTIDADE DE RESPOSTAS
    # ========================================================

    if len(df_rf) != TOTAL_RESPOSTAS_ESPERADO:
        raise ValueError(
            f"Random Forest possui {len(df_rf)} respostas. "
            f"Esperado: {TOTAL_RESPOSTAS_ESPERADO}."
        )

    if len(df_caramelo) != TOTAL_RESPOSTAS_ESPERADO:
        raise ValueError(
            f"Caramelo-Smile-2 possui "
            f"{len(df_caramelo)} respostas. "
            f"Esperado: {TOTAL_RESPOSTAS_ESPERADO}."
        )

    if len(df_tg) != TOTAL_RESPOSTAS_ESPERADO:
        raise ValueError(
            f"Arquivo de respostas possui "
            f"{len(df_tg)} respostas. "
            f"Esperado: {TOTAL_RESPOSTAS_ESPERADO}."
        )


    # ========================================================
    # PREPARA RANDOM FOREST
    # ========================================================

    df = df_tg.merge(
        df_rf,
        on="response_id",
        how="inner",
        validate="one_to_one"
    )

    if len(df) != TOTAL_RESPOSTAS_ESPERADO:
        raise ValueError(
            "A junção das respostas com o Random Forest "
            "não resultou em 300 linhas."
        )

    df["sentimento_pt"] = (
        df["rf_label"]
        .map(MAPA_SENTIMENTO)
    )

    df["categoria_pt"] = (
        df["categoria"]
        .map(MAPA_CATEGORIA)
    )

    if df["sentimento_pt"].isna().any():
        desconhecidos = (
            df.loc[
                df["sentimento_pt"].isna(),
                "rf_label"
            ]
            .unique()
        )

        raise ValueError(
            f"Rótulos desconhecidos no Random Forest: "
            f"{desconhecidos}"
        )

    if df["categoria_pt"].isna().any():
        desconhecidas = (
            df.loc[
                df["categoria_pt"].isna(),
                "categoria"
            ]
            .unique()
        )

        raise ValueError(
            f"Categorias desconhecidas: "
            f"{desconhecidas}"
        )


    # ========================================================
    # PREPARA CARAMELO-SMILE-2
    # ========================================================

    df_caramelo["sentimento_pt"] = (
        df_caramelo["classificacao"]
        .map(MAPA_SENTIMENTO)
    )

    if df_caramelo["sentimento_pt"].isna().any():

        desconhecidos = (
            df_caramelo.loc[
                df_caramelo["sentimento_pt"].isna(),
                "classificacao"
            ]
            .unique()
        )

        raise ValueError(
            "Rótulos desconhecidos no "
            f"Caramelo-Smile-2: {desconhecidos}"
        )


    # ========================================================
    # RESUMOS INICIAIS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUMO DAS CLASSIFICAÇÕES")
    print("=" * 70)

    print("\nRandom Forest + TF-IDF:")

    print(
        df["sentimento_pt"]
        .value_counts()
        .reindex(ORDEM_SENTIMENTOS)
        .fillna(0)
        .astype(int)
    )

    print("\nCaramelo-Smile-2:")

    print(
        df_caramelo["sentimento_pt"]
        .value_counts()
        .reindex(ORDEM_SENTIMENTOS)
        .fillna(0)
        .astype(int)
    )


    # ========================================================
    # GRÁFICO 1
    # RANDOM FOREST POR PROVEDOR
    # ========================================================

    print(
        "\n1. Gerando: "
        "01_sentimento_por_provedor_rf.png"
    )

    tab_qtd_prov = pd.crosstab(
        df["provedor"],
        df["sentimento_pt"]
    ).reindex(
        index=ORDEM_PROVEDORES,
        columns=ORDEM_SENTIMENTOS
    ).fillna(0)

    tab_pct_prov = (
        pd.crosstab(
            df["provedor"],
            df["sentimento_pt"],
            normalize="index"
        ) * 100
    ).reindex(
        index=ORDEM_PROVEDORES,
        columns=ORDEM_SENTIMENTOS
    ).fillna(0)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    tab_pct_prov.plot(
        kind="bar",
        ax=ax,
        color=[
            CORES_SENTIMENTO[c]
            for c in ORDEM_SENTIMENTOS
        ],
        width=0.72,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Distribuição das Classificações de Sentimento "
        "por Modelo de IA — Random Forest",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Proporção das respostas (%)",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Modelo de inteligência artificial",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(0, 105)

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Classificação",
        frameon=True
    )

    anotar_barras_agrupadas(
        ax,
        tab_qtd_prov,
        tab_pct_prov
    )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "01_sentimento_por_provedor_rf.png"
    )


    # ========================================================
    # GRÁFICO 2
    # RANDOM FOREST POR CATEGORIA DA MENSAGEM
    # ========================================================

    print(
        "\n2. Gerando: "
        "02_sentimento_por_categoria_prompt.png"
    )

    tab_qtd_cat = pd.crosstab(
        df["categoria_pt"],
        df["sentimento_pt"]
    ).reindex(
        index=ORDEM_CATEGORIAS,
        columns=ORDEM_SENTIMENTOS
    ).fillna(0)

    tab_pct_cat = (
        pd.crosstab(
            df["categoria_pt"],
            df["sentimento_pt"],
            normalize="index"
        ) * 100
    ).reindex(
        index=ORDEM_CATEGORIAS,
        columns=ORDEM_SENTIMENTOS
    ).fillna(0)

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    tab_pct_cat.plot(
        kind="bar",
        ax=ax,
        color=[
            CORES_SENTIMENTO[c]
            for c in ORDEM_SENTIMENTOS
        ],
        width=0.75,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Classificações do Random Forest "
        "por Categoria da Mensagem",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Proporção das respostas (%)",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Categoria da mensagem no conjunto de dados",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(0, 105)

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Classificação",
        frameon=True
    )

    anotar_barras_agrupadas(
        ax,
        tab_qtd_cat,
        tab_pct_cat
    )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "02_sentimento_por_categoria_prompt.png"
    )


    # ========================================================
    # COMPARAÇÃO RF x CARAMELO-SMILE-2
    # ========================================================

    df_comp = df_rf.merge(
        df_caramelo[
            [
                "response_id",
                "classificacao",
                "score",
                "sentimento_pt",
            ]
        ],
        on="response_id",
        how="inner",
        validate="one_to_one"
    )

    if len(df_comp) != TOTAL_RESPOSTAS_ESPERADO:
        raise ValueError(
            "A comparação entre Random Forest e "
            "Caramelo-Smile-2 não possui 300 respostas."
        )

    df_comp["rf_sentimento_pt"] = (
        df_comp["rf_label"]
        .map(MAPA_SENTIMENTO)
    )

    df_comp.rename(
        columns={
            "sentimento_pt":
                "caramelo_sentimento_pt",
            "score":
                "caramelo_score"
        },
        inplace=True
    )

    df_comp["concordancia"] = (
        df_comp["rf_sentimento_pt"]
        ==
        df_comp["caramelo_sentimento_pt"]
    )


    # ========================================================
    # SALVA COMPARAÇÃO INDIVIDUAL
    # ========================================================

    colunas_comparacao = [
        "response_id",
        "rf_label",
        "rf_sentimento_pt",
    ]

    if "rf_score" in df_comp.columns:
        colunas_comparacao.append(
            "rf_score"
        )

    colunas_comparacao.extend([
        "classificacao",
        "caramelo_sentimento_pt",
        "caramelo_score",
        "concordancia",
    ])

    df_comp[
        colunas_comparacao
    ].to_csv(
        COMPARACAO_CSV,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # CONCORDÂNCIA ENTRE CLASSIFICADORES
    # ========================================================

    concordantes = int(
        df_comp["concordancia"].sum()
    )

    divergentes = (
        len(df_comp) - concordantes
    )

    pct_concordancia = (
        concordantes /
        len(df_comp)
    ) * 100

    pct_divergencia = (
        divergentes /
        len(df_comp)
    ) * 100


    print("\n" + "=" * 70)
    print(
        "CONCORDÂNCIA ENTRE RANDOM FOREST "
        "E CARAMELO-SMILE-2"
    )
    print("=" * 70)

    print(
        f"Concordâncias: "
        f"{concordantes}/"
        f"{len(df_comp)} "
        f"({pct_concordancia:.2f}%)"
    )

    print(
        f"Divergências: "
        f"{divergentes}/"
        f"{len(df_comp)} "
        f"({pct_divergencia:.2f}%)"
    )

    matriz = pd.crosstab(
        df_comp["rf_sentimento_pt"],
        df_comp["caramelo_sentimento_pt"]
    ).reindex(
        index=ORDEM_SENTIMENTOS,
        columns=ORDEM_SENTIMENTOS
    ).fillna(0).astype(int)

    print(
        "\nMatriz de comparação "
        "(linhas = Random Forest; "
        "colunas = Caramelo-Smile-2):"
    )

    print(matriz)

    print(
        "\nArquivo individual salvo em:"
    )

    print(
        COMPARACAO_CSV.resolve()
    )


    # ========================================================
    # GRÁFICO 3
    # DISTRIBUIÇÃO GLOBAL DOS DOIS CLASSIFICADORES
    # ========================================================

    print(
        "\n3. Gerando: "
        "03_comparativo_rf_vs_caramelo_smile_2.png"
    )

    rf_qtd = (
        df_comp["rf_sentimento_pt"]
        .value_counts()
        .reindex(ORDEM_SENTIMENTOS)
        .fillna(0)
    )

    car_qtd = (
        df_comp["caramelo_sentimento_pt"]
        .value_counts()
        .reindex(ORDEM_SENTIMENTOS)
        .fillna(0)
    )

    rf_pct = (
        rf_qtd /
        len(df_comp)
    ) * 100

    car_pct = (
        car_qtd /
        len(df_comp)
    ) * 100

    df_comp_pct = pd.DataFrame({
        NOME_RF: rf_pct,
        NOME_CARAMELO: car_pct,
    })

    df_comp_qtd = pd.DataFrame({
        NOME_RF: rf_qtd,
        NOME_CARAMELO: car_qtd,
    })

    fig, ax = plt.subplots(
        figsize=(8.8, 5.5)
    )

    df_comp_pct.plot(
        kind="bar",
        ax=ax,
        color=[
            "#3498db",
            "#9b59b6",
        ],
        width=0.62,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Distribuição das Classificações de Sentimento: "
        "Random Forest e Caramelo-Smile-2",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Proporção das respostas (%)",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Classificação de sentimento",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(0, 105)

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Classificador",
        frameon=True
    )

    anotar_barras_agrupadas(
        ax,
        df_comp_qtd,
        df_comp_pct
    )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "03_comparativo_rf_vs_caramelo_smile_2.png"
    )


    # ========================================================
    # PROCESSAMENTO DA AVALIAÇÃO HUMANA
    # ========================================================

    print(
        "\nProcessando avaliação humana..."
    )

    df_hum = pd.read_csv(
        HUMAN_CSV,
        encoding="utf-8-sig"
    )

    with open(
        CHAVE_JSON,
        "r",
        encoding="utf-8"
    ) as arquivo:

        chave = json.load(arquivo)


    # Posições originais das quatro mensagens
    # selecionadas para a avaliação humana.
    blocos = [
        ("M011", "Depressão", 7),
        ("M028", "Solidão", 17),
        ("M052", "Ansiedade", 27),
        ("M077", "Estresse", 37),
    ]


    preferencias = []
    criterios = []


    for (
        msg_id,
        categoria_nome,
        coluna_inicial
    ) in blocos:

        if msg_id not in chave:
            raise KeyError(
                f"{msg_id} não foi encontrado "
                "na chave de cegamento."
            )

        mapa_modelos = chave[msg_id]

        for i, alias in enumerate([
            "Modelo A",
            "Modelo B",
            "Modelo C"
        ]):

            if alias not in mapa_modelos:
                raise KeyError(
                    f"{alias} não encontrado "
                    f"para {msg_id}."
                )

            provedor_real = (
                mapa_modelos[alias]
            )

            coluna_empatia = (
                coluna_inicial +
                (i * 3)
            )

            coluna_seguranca = (
                coluna_inicial +
                (i * 3) + 1
            )

            coluna_naturalidade = (
                coluna_inicial +
                (i * 3) + 2
            )

            for (
                idx_participante,
                linha
            ) in df_hum.iterrows():

                criterios.append({
                    "participante":
                        idx_participante + 1,

                    "mensagem":
                        msg_id,

                    "categoria":
                        categoria_nome,

                    "provedor":
                        provedor_real,

                    "Empatia":
                        converter_likert(
                            linha.iloc[
                                coluna_empatia
                            ]
                        ),

                    "Segurança":
                        converter_likert(
                            linha.iloc[
                                coluna_seguranca
                            ]
                        ),

                    "Naturalidade":
                        converter_likert(
                            linha.iloc[
                                coluna_naturalidade
                            ]
                        ),
                })


        coluna_preferencia = (
            coluna_inicial + 9
        )

        for (
            idx_participante,
            linha
        ) in df_hum.iterrows():

            escolha_alias = (
                linha.iloc[
                    coluna_preferencia
                ]
            )

            provedor_escolhido = (
                mapa_modelos.get(
                    escolha_alias,
                    escolha_alias
                )
            )

            preferencias.append({
                "participante":
                    idx_participante + 1,

                "mensagem":
                    msg_id,

                "categoria":
                    categoria_nome,

                "provedor_escolhido":
                    provedor_escolhido,
            })


    df_crit = pd.DataFrame(
        criterios
    )

    df_pref = pd.DataFrame(
        preferencias
    )


    # ========================================================
    # CONFERE VALORES DA ESCALA LIKERT
    # ========================================================

    colunas_likert = [
        "Empatia",
        "Segurança",
        "Naturalidade",
    ]

    if (
        df_crit[
            colunas_likert
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Existem respostas da escala Likert "
            "que não puderam ser convertidas. "
            "Confira o CSV do formulário."
        )


    # ========================================================
    # GRÁFICO 4
    # PREFERÊNCIA HUMANA GERAL
    # ========================================================

    print(
        "\n4. Gerando: "
        "04_avaliacao_humana_preferencia_geral.png"
    )

    votos_qtd = (
        df_pref[
            "provedor_escolhido"
        ]
        .value_counts()
        .reindex(
            ORDEM_PROVEDORES
        )
        .fillna(0)
    )

    votos_pct = (
        votos_qtd /
        len(df_pref)
    ) * 100

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    barras = ax.bar(
        ORDEM_PROVEDORES,
        votos_pct,
        color=[
            CORES_PROVEDORES[p]
            for p in ORDEM_PROVEDORES
        ],
        width=0.55,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Modelo Escolhido como Resposta Mais Adequada "
        "na Avaliação Humana",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Proporção das escolhas (%)",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Modelo de inteligência artificial",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        max(votos_pct) + 18
    )

    labels = []

    for i in range(
        len(ORDEM_PROVEDORES)
    ):

        labels.append(
            f"{int(votos_qtd.iloc[i])} escolhas\n"
            f"({votos_pct.iloc[i]:.1f}%)"
        )

    ax.bar_label(
        barras,
        labels=labels,
        padding=4,
        fontsize=9,
        fontweight="bold",
        color="#2c3e50"
    )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "04_avaliacao_humana_preferencia_geral.png"
    )


    # ========================================================
    # GRÁFICO 5
    # MÉDIAS DA ESCALA LIKERT
    # ========================================================

    print(
        "\n5. Gerando: "
        "05_avaliacao_humana_criterios_likert.png"
    )

    medias_crit = (
        df_crit
        .groupby("provedor")[
            [
                "Empatia",
                "Segurança",
                "Naturalidade",
            ]
        ]
        .mean()
        .reindex(
            ORDEM_PROVEDORES
        )
        .round(2)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    medias_crit.plot(
        kind="bar",
        ax=ax,
        color=[
            "#e91e63",
            "#3f51b5",
            "#009688",
        ],
        width=0.68,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Avaliação Humana por Critério "
        "(Escala Likert de 1 a 5)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Média das avaliações",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Modelo de inteligência artificial",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(
        1,
        5.2
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Critério avaliado",
        frameon=True
    )

    for (
        coluna_idx,
        container
    ) in enumerate(
        ax.containers
    ):

        labels = []

        for linha_idx, _ in enumerate(
            container
        ):

            valor = (
                medias_crit.iloc[
                    linha_idx,
                    coluna_idx
                ]
            )

            labels.append(
                f"{valor:.2f}"
            )

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=8.5,
            fontweight="bold",
            color="#2c3e50"
        )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "05_avaliacao_humana_criterios_likert.png"
    )


    # ========================================================
    # GRÁFICO 6
    # PREFERÊNCIA NOS QUATRO CENÁRIOS AVALIADOS
    # ========================================================

    print(
        "\n6. Gerando: "
        "06_preferencia_humana_por_categoria.png"
    )

    tab_votos_cat = (
        pd.crosstab(
            df_pref["categoria"],
            df_pref[
                "provedor_escolhido"
            ]
        )
        .reindex(
            index=ORDEM_CATEGORIAS,
            columns=ORDEM_PROVEDORES
        )
        .fillna(0)
    )

    tab_votos_pct = (
        pd.crosstab(
            df_pref["categoria"],
            df_pref[
                "provedor_escolhido"
            ],
            normalize="index"
        ) * 100
    ).reindex(
        index=ORDEM_CATEGORIAS,
        columns=ORDEM_PROVEDORES
    ).fillna(0)

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    tab_votos_pct.plot(
        kind="bar",
        ax=ax,
        color=[
            CORES_PROVEDORES[p]
            for p in ORDEM_PROVEDORES
        ],
        width=0.72,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title(
        "Preferência Humana nos Quatro "
        "Cenários Avaliados",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_ylabel(
        "Proporção das escolhas (%)",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Categoria da mensagem selecionada",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Modelo escolhido",
        frameon=True
    )

    anotar_barras_agrupadas(
        ax,
        tab_votos_cat,
        tab_votos_pct
    )

    salvar_figura(
        fig,
        DIR_OUTPUT /
        "06_preferencia_humana_por_categoria.png"
    )


    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("GRÁFICOS GERADOS COM SUCESSO")
    print("=" * 70)

    print(
        "Pasta dos gráficos:"
    )

    print(
        DIR_OUTPUT.resolve()
    )

    print()

    print(
        "Arquivo de comparação RF x "
        "Caramelo-Smile-2:"
    )

    print(
        COMPARACAO_CSV.resolve()
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    gerar_graficos()