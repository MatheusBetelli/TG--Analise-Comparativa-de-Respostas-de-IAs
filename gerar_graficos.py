from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# Estilo e configurações visuais
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

DIR_OUTPUT = Path("data/processed/graficos")
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

RF_CSV = Path("data/processed/resultados_random_forest.csv")
CARAMELO_CSV = Path("data/processed/resultados_caramelo.csv")
TG_CSV = Path("data/output/respostas_ias_tg.csv")
HUMAN_CSV = Path("data/output/respostas_formulario_humanos.csv")
CHAVE_JSON = Path("data/processed/chave_cegamento.json")

# Cores padronizadas por provedor de IA
CORES_PROVEDORES = {
    "OpenAI": "#10a37f",     # Verde OpenAI
    "Anthropic": "#d97706",  # Laranja/Âmbar Claude
    "Meta": "#0284c7"        # Azul Meta
}

# Cores padronizadas para sentimentos
CORES_SENTIMENTO = {
    "Positivo": "#2ecc71",   # Verde
    "Neutro": "#95a5a6",     # Cinza
    "Negativo": "#e74c3c"    # Vermelho
}


def gerar_graficos():
    print("=" * 65)
    print("GERANDO GRÁFICOS DO EXPERIMENTO COMPLETO DO TG")
    print("Incluindo Análise de IA (RF, Caramelo) e Avaliação Humana")
    print("=" * 65)

    if not RF_CSV.exists() or not TG_CSV.exists():
        raise FileNotFoundError("Arquivos necessários não encontrados!")

    df_rf = pd.read_csv(RF_CSV, encoding="utf-8-sig")
    df_tg = pd.read_csv(TG_CSV, encoding="utf-8-sig")
    df = df_tg.merge(df_rf, on="response_id")

    mapa_sentimento = {
        "POSITIVO": "Positivo",
        "NEUTRO": "Neutro",
        "NEGATIVO": "Negativo",
        "positive": "Positivo",
        "neutral": "Neutro",
        "negative": "Negativo"
    }

    mapa_categoria = {
        "Depression": "Depressão",
        "Loneliness": "Solidão",
        "Anxiety": "Ansiedade",
        "Stress": "Estresse"
    }

    df["sentimento_pt"] = df["rf_label"].map(mapa_sentimento)
    df["categoria_pt"] = df["categoria"].map(mapa_categoria)

    colunas_sentimento = ["Positivo", "Neutro", "Negativo"]
    cores_lista = [CORES_SENTIMENTO[c] for c in colunas_sentimento]

    # -------------------------------------------------------------
    # Gráfico 1: Sentimento por Provedor de IA (Random Forest)
    # -------------------------------------------------------------
    print("\n1. Gerando: 01_sentimento_por_provedor_rf.png")
    tab_qtd_prov = pd.crosstab(df["provedor"], df["sentimento_pt"]).reindex(columns=colunas_sentimento).fillna(0)
    tab_pct_prov = (pd.crosstab(df["provedor"], df["sentimento_pt"], normalize="index") * 100).reindex(columns=colunas_sentimento).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    tab_pct_prov.plot(
        kind="bar",
        ax=ax,
        color=cores_lista,
        width=0.72,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title("Distribuição de Sentimento por Modelo de IA (Random Forest + TF-IDF)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Proporção das Respostas (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Modelo de Inteligência Artificial", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10, fontweight="bold")
    ax.legend(title="Sentimento", frameon=True, loc="upper right")

    num_cols = len(colunas_sentimento)
    num_rows = len(tab_pct_prov)
    for c_idx in range(num_cols):
        for r_idx in range(num_rows):
            patch_idx = c_idx * num_rows + r_idx
            p = ax.patches[patch_idx]
            qtd = int(tab_qtd_prov.iloc[r_idx, c_idx])
            pct = tab_pct_prov.iloc[r_idx, c_idx]
            altura = p.get_height()

            ax.annotate(
                f"{qtd}\n({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2., altura + 2),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50"
            )

    plt.tight_layout()
    caminho_g1 = DIR_OUTPUT / "01_sentimento_por_provedor_rf.png"
    plt.savefig(caminho_g1)
    plt.close()
    print(f"Salvo: {caminho_g1}")

    # -------------------------------------------------------------
    # Gráfico 2: Sentimento por Categoria de Sofrimento do Prompt
    # -------------------------------------------------------------
    print("\n2. Gerando: 02_sentimento_por_categoria_prompt.png")
    ordem_categorias = ["Depressão", "Solidão", "Ansiedade", "Estresse"]
    tab_qtd_cat = pd.crosstab(df["categoria_pt"], df["sentimento_pt"]).reindex(index=ordem_categorias, columns=colunas_sentimento).fillna(0)
    tab_pct_cat = (pd.crosstab(df["categoria_pt"], df["sentimento_pt"], normalize="index") * 100).reindex(index=ordem_categorias, columns=colunas_sentimento).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    tab_pct_cat.plot(
        kind="bar",
        ax=ax,
        color=cores_lista,
        width=0.75,
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title("Sentimento das Respostas por Categoria do Prompt (Random Forest)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Proporção (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Vulnerabilidade Emocional do Usuário (Prompt)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10, fontweight="bold")
    ax.legend(title="Sentimento", frameon=True, loc="upper right")

    num_rows_cat = len(tab_pct_cat)
    for c_idx in range(num_cols):
        for r_idx in range(num_rows_cat):
            patch_idx = c_idx * num_rows_cat + r_idx
            p = ax.patches[patch_idx]
            qtd = int(tab_qtd_cat.iloc[r_idx, c_idx])
            pct = tab_pct_cat.iloc[r_idx, c_idx]
            altura = p.get_height()

            ax.annotate(
                f"{qtd}\n({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2., altura + 2),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#2c3e50"
            )

    plt.tight_layout()
    caminho_g2 = DIR_OUTPUT / "02_sentimento_por_categoria_prompt.png"
    plt.savefig(caminho_g2)
    plt.close()
    print(f"Salvo: {caminho_g2}")

    # -------------------------------------------------------------
    # Gráfico 3: Comparativo Random Forest vs Caramelo-Smile
    # -------------------------------------------------------------
    if CARAMELO_CSV.exists():
        print("\n3. Gerando: 03_comparativo_rf_vs_caramelo.png")
        df_caramelo = pd.read_csv(CARAMELO_CSV, encoding="utf-8-sig")
        df_caramelo["sentimento_pt"] = df_caramelo["classificacao"].map(mapa_sentimento)

        df_comp = df_rf.merge(df_caramelo, on="response_id", suffixes=("_rf", "_caramelo"))

        rf_qtd = df_comp["rf_label"].map(mapa_sentimento).value_counts().reindex(colunas_sentimento).fillna(0)
        car_qtd = df_comp["sentimento_pt"].value_counts().reindex(colunas_sentimento).fillna(0)

        rf_pct = (rf_qtd / len(df_comp)) * 100
        car_pct = (car_qtd / len(df_comp)) * 100

        df_comp_pct = pd.DataFrame({
            "Random Forest + TF-IDF": rf_pct,
            "Caramelo-Smile (BERT)": car_pct
        })
        df_comp_qtd = pd.DataFrame({
            "Random Forest + TF-IDF": rf_qtd,
            "Caramelo-Smile (BERT)": car_qtd
        })

        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        df_comp_pct.plot(
            kind="bar",
            ax=ax,
            color=["#3498db", "#9b59b6"],
            width=0.62,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title("Comparação Global: Random Forest vs Caramelo-Smile", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Percentual das Respostas (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sentimento Classificado", fontsize=11, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10, fontweight="bold")
        ax.legend(title="Modelo", frameon=True, loc="upper right")
        ax.set_ylim(0, 118)

        num_cols_comp = 2
        num_rows_comp = len(colunas_sentimento)
        for c_idx in range(num_cols_comp):
            for r_idx in range(num_rows_comp):
                patch_idx = c_idx * num_rows_comp + r_idx
                p = ax.patches[patch_idx]
                qtd = int(df_comp_qtd.iloc[r_idx, c_idx])
                pct = df_comp_pct.iloc[r_idx, c_idx]
                altura = p.get_height()

                ax.annotate(
                    f"{qtd}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., altura + 2),
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#2c3e50"
                )

        plt.tight_layout()
        caminho_g3 = DIR_OUTPUT / "03_comparativo_rf_vs_caramelo.png"
        plt.savefig(caminho_g3)
        plt.close()
        print(f"Salvo: {caminho_g3}")

    # =============================================================
    # PROCESSAMENTO DO FORMULÁRIO HUMANO
    # =============================================================
    if HUMAN_CSV.exists() and CHAVE_JSON.exists():
        print("\nProcessando respostas do formulário humano...")
        df_hum = pd.read_csv(HUMAN_CSV)
        with open(CHAVE_JSON, "r", encoding="utf-8") as f:
            chave = json.load(f)

        mapa_likert = {
            "1 — Discordo totalmente": 1,
            "2 — Discordo": 2,
            "3 — Neutro / parcialmente": 3,
            "4 — Concordo": 4,
            "5 — Concordo totalmente": 5
        }

        blocos = [
            ("M011", "Depressão", 7),
            ("M028", "Solidão", 17),
            ("M052", "Ansiedade", 27),
            ("M077", "Estresse", 37)
        ]

        preferencias = []
        criterios = []

        for msg_id, cat_nome, start_col in blocos:
            map_modelos = chave[msg_id]
            for i, mod_alias in enumerate(["Modelo A", "Modelo B", "Modelo C"]):
                prov_real = map_modelos[mod_alias]
                col_emp = start_col + (i * 3)
                col_seg = start_col + (i * 3) + 1
                col_nat = start_col + (i * 3) + 2

                for idx_part, row in df_hum.iterrows():
                    criterios.append({
                        "participante": idx_part + 1,
                        "bloco": msg_id,
                        "categoria": cat_nome,
                        "provedor": prov_real,
                        "Empatia": mapa_likert.get(row.iloc[col_emp]),
                        "Segurança": mapa_likert.get(row.iloc[col_seg]),
                        "Naturalidade": mapa_likert.get(row.iloc[col_nat]),
                    })

            col_pref = start_col + 9
            for idx_part, row in df_hum.iterrows():
                escolha_alias = row.iloc[col_pref]
                prov_escolhido = map_modelos.get(escolha_alias, escolha_alias)
                preferencias.append({
                    "participante": idx_part + 1,
                    "bloco": msg_id,
                    "categoria": cat_nome,
                    "provedor_escolhido": prov_escolhido
                })

        df_crit = pd.DataFrame(criterios)
        df_pref = pd.DataFrame(preferencias)

        ordem_provedores = ["OpenAI", "Meta", "Anthropic"]
        cores_prov_lista = [CORES_PROVEDORES[p] for p in ordem_provedores]

        # ---------------------------------------------------------
        # Gráfico 4: Avaliação Humana - Preferência Geral (Votos)
        # ---------------------------------------------------------
        print("\n4. Gerando: 04_avaliacao_humana_preferencia_geral.png")
        votos_qtd = df_pref["provedor_escolhido"].value_counts().reindex(ordem_provedores).fillna(0)
        votos_pct = (votos_qtd / len(df_pref)) * 100

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        barras = ax.bar(
            ordem_provedores,
            votos_pct,
            color=cores_prov_lista,
            width=0.55,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title("Preferência Humana: Modelo Escolhido como MAIS ADEQUADO", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Percentual de Votos (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Modelo de Inteligência Artificial", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(votos_pct) + 18)
        ax.set_xticks(range(len(ordem_provedores)))
        ax.set_xticklabels(ordem_provedores, fontsize=10, fontweight="bold")

        for idx, p in enumerate(barras):
            altura = p.get_height()
            qtd = int(votos_qtd.iloc[idx])
            ax.annotate(
                f"{qtd} votos\n({altura:.1f}%)",
                (p.get_x() + p.get_width() / 2., altura + 1.5),
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#2c3e50"
            )

        plt.tight_layout()
        caminho_g4 = DIR_OUTPUT / "04_avaliacao_humana_preferencia_geral.png"
        plt.savefig(caminho_g4)
        plt.close()
        print(f"Salvo: {caminho_g4}")

        # ---------------------------------------------------------
        # Gráfico 5: Médias nos Critérios Likert (Empatia, Segurança, Naturalidade)
        # ---------------------------------------------------------
        print("\n5. Gerando: 05_avaliacao_humana_criterios_likert.png")
        medias_crit = df_crit.groupby("provedor")[["Empatia", "Segurança", "Naturalidade"]].mean().reindex(ordem_provedores).round(2)

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        cores_criterios = ["#e91e63", "#3f51b5", "#009688"]  # Rosa (Empatia), Azul (Segurança), Verde-água (Naturalidade)

        medias_crit.plot(
            kind="bar",
            ax=ax,
            color=cores_criterios,
            width=0.68,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title("Avaliação Humana por Critérios (Média na Escala Likert de 1 a 5)", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Nota Média (1 = Discordo Totalmente, 5 = Concordo Totalmente)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Modelo de Inteligência Artificial", fontsize=11, fontweight="bold")
        ax.set_ylim(1, 5.2)
        ax.set_xticklabels(ordem_provedores, rotation=0, fontsize=10, fontweight="bold")
        ax.legend(title="Critério Avaliado", frameon=True, loc="upper right")

        for c_idx in range(3):
            for r_idx in range(len(ordem_provedores)):
                patch_idx = c_idx * len(ordem_provedores) + r_idx
                p = ax.patches[patch_idx]
                valor = medias_crit.iloc[r_idx, c_idx]
                ax.annotate(
                    f"{valor:.2f}",
                    (p.get_x() + p.get_width() / 2., valor + 0.08),
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#2c3e50"
                )

        plt.tight_layout()
        caminho_g5 = DIR_OUTPUT / "05_avaliacao_humana_criterios_likert.png"
        plt.savefig(caminho_g5)
        plt.close()
        print(f"Salvo: {caminho_g5}")

        # ---------------------------------------------------------
        # Gráfico 6: Preferência Humana por Categoria do Desabafo
        # ---------------------------------------------------------
        print("\n6. Gerando: 06_preferencia_humana_por_categoria.png")
        tab_votos_cat = pd.crosstab(df_pref["categoria"], df_pref["provedor_escolhido"]).reindex(index=ordem_categorias, columns=ordem_provedores).fillna(0)
        tab_votos_pct = (pd.crosstab(df_pref["categoria"], df_pref["provedor_escolhido"], normalize="index") * 100).reindex(index=ordem_categorias, columns=ordem_provedores).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        tab_votos_pct.plot(
            kind="bar",
            ax=ax,
            color=cores_prov_lista,
            width=0.72,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title("Preferência Humana da Resposta Mais Adequada por Categoria", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Proporção dos Votos (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Categoria de Sofrimento do Usuário", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.set_xticklabels(ordem_categorias, rotation=0, fontsize=10, fontweight="bold")
        ax.legend(title="Modelo Escolhido", frameon=True, loc="upper right")

        for c_idx in range(len(ordem_provedores)):
            for r_idx in range(len(ordem_categorias)):
                patch_idx = c_idx * len(ordem_categorias) + r_idx
                p = ax.patches[patch_idx]
                qtd = int(tab_votos_cat.iloc[r_idx, c_idx])
                pct = tab_votos_pct.iloc[r_idx, c_idx]
                altura = p.get_height()

                ax.annotate(
                    f"{qtd} votos\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., altura + 2),
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#2c3e50"
                )

        plt.tight_layout()
        caminho_g6 = DIR_OUTPUT / "06_preferencia_humana_por_categoria.png"
        plt.savefig(caminho_g6)
        plt.close()
        print(f"Salvo: {caminho_g6}")

    print("\n" + "=" * 65)
    print("TODOS OS 6 GRÁFICOS GERADOS COM SUCESSO!")
    print("Pasta de destino:", DIR_OUTPUT.resolve())
    print("=" * 65)


if __name__ == "__main__":
    gerar_graficos()
