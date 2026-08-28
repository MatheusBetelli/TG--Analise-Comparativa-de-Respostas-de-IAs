import json
import random
import pandas as pd
from pathlib import Path

# Definir semente aleatória para reprodutibilidade
random.seed(42)

INPUT_CSV = Path("data/output/respostas_ias_tg.csv")
if not INPUT_CSV.exists():
    INPUT_CSV = Path("data/output/respostas_ias.csv")

OUTPUT_TXT = Path("data/output/formulario_pronto.txt")
OUTPUT_KEY_JSON = Path("data/processed/chave_cegamento.json")
OUTPUT_CSV_BLIND = Path("data/output/dataset_humano_cegado.csv")

# Mapeamento dos 4 prompts leves e seus textos polidos em Português fluente
PROMPTS_LEVES = {
    'Depression': {
        'id': 'M011',
        'texto_polido': 'Um amigo me disse ontem: "Como você pode estar desanimado se está sempre rindo?". A verdade é que quando estou com outras pessoas, acabo fingindo que está tudo bem para não preocupar ninguém. Mas por dentro me sinto super esgotado.'
    },
    'Loneliness': {
        'id': 'M028',
        'texto_polido': 'Quando me comparo com os outros, não entendo por que é tão difícil para mim fazer novos amigos ou ter relacionamentos próximos. Às vezes parece que todo mundo consegue se conectar facilmente, menos eu.'
    },
    'Anxiety': {
        'id': 'M052',
        'texto_polido': 'Mais alguém tem o dilema do café? Fico sempre na dúvida entre tomar café e a ansiedade piorar, ou não tomar e passar o dia inteiro exausto sem conseguir focar.'
    },
    'Stress': {
        'id': 'M077',
        'texto_polido': 'Como vocês lidam com a ansiedade sobre um compromisso ou prazo importante que está chegando? Toda vez que penso nisso me dá uma tensão forte e uma leve dor de cabeça.'
    }
}

def gerar_formulario_humanos():
    print("=" * 65)
    print("GERADOR DE FORMULÁRIO HUMANO (PROMPTS POLIDOS E NATURAIS)")
    print("=" * 65)
    
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo {INPUT_CSV} não encontrado!")
        
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"Dataset carregado: {len(df)} respostas.")
    
    prompts_ids = [v['id'] for v in PROMPTS_LEVES.values()]
    print(f"Prompts selecionados (M011, M028, M052, M077): {prompts_ids}")
    
    df_selecionado = df[df['mensagem_id'].isin(prompts_ids)].copy()
    
    chave_secreta = {}
    blocos_formulario_txt = []
    registros_cegados_csv = []
    
    label_alias = ['Modelo A', 'Modelo B', 'Modelo C']
    
    blocos_formulario_txt.append("=" * 70)
    blocos_formulario_txt.append("  FORMULÁRIO DE AVALIAÇÃO HUMANA — TEXTOS POLIDOS E NATURAIS")
    blocos_formulario_txt.append("=" * 70)
    blocos_formulario_txt.append("Instruções: Copie a [MENSAGEM DO USUÁRIO] e as opções de [RESPOSTAS ANONIMIZADAS].\n")
    
    for i, (cat_nome, info) in enumerate(PROMPTS_LEVES.items(), 1):
        msg_id = info['id']
        mensagem_prompt = info['texto_polido']
        
        df_prompt = df_selecionado[df_selecionado['mensagem_id'] == msg_id].copy()
        
        # Embaralhar a ordem das IAs para este prompt específico
        provedores = list(df_prompt['provedor'].unique())
        random.shuffle(provedores)
        
        # Mapeamento do cegamento para este prompt
        mapeamento_prompt = {}
        for alias, prov in zip(label_alias, provedores):
            mapeamento_prompt[alias] = prov
            
        chave_secreta[msg_id] = mapeamento_prompt
        
        # Montar texto do bloco para o Forms
        bloco_str = f"--- BLOCO {i:02d}/4 | Categoria: {cat_nome} (ID: {msg_id}) ---\n\n"
        bloco_str += f"💬 MENSAGEM DO USUÁRIO:\n\"{mensagem_prompt}\"\n\n"
        bloco_str += "🤖 RESPOSTAS PARA AVALIAR:\n\n"
        
        for alias in label_alias:
            prov_real = mapeamento_prompt[alias]
            resp_row = df_prompt[df_prompt['provedor'] == prov_real].iloc[0]
            texto_resp = resp_row['resposta']
            
            bloco_str += f"[{alias}]:\n\"{texto_resp}\"\n\n"
            
            registros_cegados_csv.append({
                "bloco_num": i,
                "mensagem_id": msg_id,
                "categoria": cat_nome,
                "mensagem": mensagem_prompt,
                "modelo_cegado": alias,
                "provedor_real": prov_real,
                "resposta": texto_resp
            })
            
        bloco_str += "-" * 70 + "\n"
        blocos_formulario_txt.append(bloco_str)
        
    # Salvar Chave Secreta JSON
    OUTPUT_KEY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_KEY_JSON, "w", encoding="utf-8") as f:
        json.dump(chave_secreta, f, ensure_ascii=False, indent=2)
        
    # Salvar Arquivo de Texto do Formulário
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(blocos_formulario_txt))
        
    # Salvar CSV Cegado
    df_cegado_csv = pd.DataFrame(registros_cegados_csv)
    df_cegado_csv.to_csv(OUTPUT_CSV_BLIND, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 65)
    print("SUCESSO! ARQUIVOS POLIDOS GERADOS COM ÊXITO:")
    print("=" * 65)
    print(f"1. Formulário em Texto Polido: {OUTPUT_TXT.resolve()}")
    print(f"2. Chave Secreta (JSON):      {OUTPUT_KEY_JSON.resolve()}")

if __name__ == "__main__":
    gerar_formulario_humanos()
