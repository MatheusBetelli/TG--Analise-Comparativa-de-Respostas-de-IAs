# TG — Respostas de LLMs em contextos de vulnerabilidade emocional

## Estrutura atual do experimento

O projeto parte de um dataset público com postagens reais do Reddit relacionadas
a saúde mental e vulnerabilidade emocional.

Foram definidas quatro categorias para o experimento:

- Depression
- Loneliness
- Anxiety
- Stress

São selecionadas 25 mensagens de cada categoria, totalizando 100 mensagens.

As mensagens originais em inglês são preservadas e traduzidas para português.
Depois, as mesmas 100 mensagens em português são enviadas para três modelos de
linguagem via OpenRouter:

- OpenAI
- Anthropic
- Meta

O resultado esperado é de 300 respostas.

Essas respostas são analisadas posteriormente pelo Caramelo-Smile, pelo
Random Forest + TF-IDF e pela avaliação humana.

---

## Estrutura de pastas

```text
TG_reorganizado/
│
├── preparar_dataset.py
├── traduzir_dataset.py
├── coletar_respostas.py
├── aplicar_caramelo.py
├── treino_rf.py
├── teste_caramelo.py
├── requirements.txt
├── configuracao_experimento.json
├── .env.example
├── .gitignore
│
└── data/
    ├── raw/
    │   └── mental_health_padronizado.csv
    │
    ├── input/
    │   ├── mensagens_tg_selecionadas.csv
    │   └── mensagens_tg.csv
    │
    ├── output/
    │   └── respostas_ias_tg.csv
    │
    └── processed/
        ├── resultados_caramelo.csv
        └── resultados_random_forest.csv
```

`mensagens_tg.csv`, `respostas_ias_tg.csv` e os arquivos de `processed` são
criados pelos scripts.

---

## 1. Criar ambiente Python

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. Preparar o .env

Copie:

```powershell
Copy-Item .env.example .env
```

Abra o `.env` e coloque uma chave válida do OpenRouter.

Não envie `.env` para o GitHub.

---

## 3. Seleção das 100 mensagens

O arquivo `data/input/mensagens_tg_selecionadas.csv` já acompanha este pacote.

Caso queira reproduzir a seleção a partir do dataset original:

```powershell
python preparar_dataset.py
```

O script:

1. remove mensagens vazias;
2. remove duplicadas;
3. mantém Depression, Loneliness, Anxiety e Stress;
4. mantém mensagens entre 20 e 200 palavras;
5. sorteia 25 de cada categoria usando `random_state=42`.

---

## 4. Traduzir inglês para português

Execute:

```powershell
python traduzir_dataset.py
```

Será criado:

```text
data/input/mensagens_tg.csv
```

Esse arquivo preserva a mensagem original e adiciona `mensagem_pt`.

Antes da coleta oficial, revise manualmente uma amostra das traduções.

---

## 5. Testar a coleta

Uma mensagem apenas com OpenAI:

```powershell
python coletar_respostas.py --limit 1 --provider OpenAI
```

Uma mensagem com cada um dos três modelos:

```powershell
python coletar_respostas.py --limit 1
```

---

## 6. Coleta oficial

Depois de conferir a tradução, modelos e parâmetros:

```powershell
python coletar_respostas.py
```

O arquivo será salvo em:

```text
data/output/respostas_ias_tg.csv
```

Com 100 mensagens e 3 modelos, o esperado é 300 respostas.

O script não repete respostas que já estejam com status `OK`.

---

## 7. Caramelo-Smile

Teste simples:

```powershell
python teste_caramelo.py
```

Análise das respostas:

```powershell
python aplicar_caramelo.py
```

Saída:

```text
data/processed/resultados_caramelo.csv
```

Colunas:

```text
response_id,classificacao,score
```

---

## 8. Random Forest

Execute:

```powershell
python treino_rf.py
```

O script treina a abordagem Random Forest + TF-IDF usando uma amostra do
B2W-Reviews01 e depois classifica as respostas do TG.

Saída:

```text
data/processed/resultados_random_forest.csv
```

---

## Ordem correta

```text
mental_health_padronizado.csv
        ↓
preparar_dataset.py
        ↓
mensagens_tg_selecionadas.csv
        ↓
traduzir_dataset.py
        ↓
mensagens_tg.csv
        ↓
coletar_respostas.py
        ↓
respostas_ias_tg.csv
        ↓
  ┌───────────────┐
  │               │
aplicar_caramelo  treino_rf
  │               │
  ↓               ↓
resultados_       resultados_
caramelo.csv      random_forest.csv
```

## Observação metodológica importante

O dataset original e a mensagem original em inglês devem ser preservados.
A versão em português é uma transformação utilizada para padronizar o idioma
do experimento. Antes de utilizar os dados em formulários ou publicar arquivos,
verifique a licença do dataset e remova informações que possam identificar
usuários.
