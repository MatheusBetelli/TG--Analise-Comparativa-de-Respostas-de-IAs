# TG — Respostas de LLMs em vulnerabilidade emocional

## Escopo
O projeto envia 100 mensagens fictícias de cunho emocional para três modelos, via OpenRouter, e salva as respostas para posterior comparação em três eixos: avaliação humana; Random Forest + TF-IDF; e análise de PLN com Caramelo-Smile.

## Estrutura
```text
tg/
├── coletar_respostas.py
├── .env.example
├── requirements.txt
├── configuracao_experimento.json
└── data/
    ├── input/
    │   ├── prompts_vulnerabilidade.csv
    │   └── modelo_avaliacao_humana.csv
    └── output/
        └── respostas_ias.csv   # criado após executar
```

## Dataset
100 prompts: 20 de Tristeza, 20 de Solidão, 20 de Ansiedade, 20 de Frustração e 20 de Estresse. As intensidades Leve/Moderada/Alta são apenas metadados experimentais e não diagnósticos.

## Instalação — Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```
Abra `.env` e coloque sua chave do OpenRouter.

## Testar um modelo
```powershell
python coletar_respostas.py --limit 1 --provider OpenAI
python coletar_respostas.py --limit 1 --provider Google
python coletar_respostas.py --limit 1 --provider Anthropic
```

## Testar os três
```powershell
python coletar_respostas.py --limit 2
```
Isso gera até 6 respostas.

## Coleta oficial
Somente após conferir os modelos e congelar a configuração:
```powershell
python coletar_respostas.py
```
São 100 prompts x 3 modelos = 300 chamadas.

## Regras metodológicas
- Mesmo prompt para todos os modelos.
- Sem system prompt adicional.
- Não editar respostas recebidas.
- Não trocar modelo durante a coleta oficial.
- Registrar IDs dos modelos, data e parâmetros.
- Guardar o CSV bruto intacto.
- Ocultar o nome do modelo na avaliação humana para reduzir viés.
- Antes do Random Forest, definir claramente qual será o rótulo-alvo.

## Saída
`data/output/respostas_ias.csv` registra response_id, prompt_id, categoria, intensidade, mensagem, provedor, modelo, resposta, tokens, tempo, status, erro e data UTC.
