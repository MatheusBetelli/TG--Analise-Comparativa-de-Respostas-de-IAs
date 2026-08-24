import os
import json
import random
import pandas as pd
from pathlib import Path

# Definir semente aleatória para reprodutibilidade
random.seed(42)

INPUT_CSV = Path("data/output/respostas_ias_tg.csv")
if not INPUT_CSV.exists():
    INPUT_CSV = Path("data/output/respostas_ias_tg.csv")

OUTPUT_TXT = Path("data/output/formulario_pronto.txt")
OUTPUT_KEY_JSON = Path("data/processed/chave_cegamento.json")
OUTPUT_CSV_BLIND = Path("data/output/dataset_humano_cegado.csv")

def gerar_formulario_humanos():
    print("=" * 65)
    print("GERADOR DE FORMULÁRIO HUMANO & CEGAMENTO (BLINDING)")
    print("=" * 65)
    
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo {INPUT_CSV} não encontrado!")
        
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"Dataset carregado: {len(df)} respostas.")
    
    # 1. Obter lista única de categorias e prompts
    categorias = df['categoria'].unique()
    print(f"Categorias encontradas ({len(categorias)}): {list(categorias)}")
    
    # 2. Selecionar 4 prompts por categoria (totalizando 20 prompts)
    prompts_selecionados = []
    for cat in categorias:
        prompts_cat = df[df['categoria'] == cat]['prompt_id'].unique()
        # Selecionar até 4 prompts por categoria
        escolhidos = list(prompts_cat[:4])
        prompts_selecionados.extend(escolhidos)
        
    print(f"Total de Prompts Selecionados para o Formulário: {len(prompts_selecionados)} (4 por categoria)")
    
    # Filtrar o DataFrame apenas com os 20 prompts escolhidos
    df_selecionado = df[df['prompt_id'].isin(prompts_selecionados)].copy()
    
    # 3. Processar cegamento e criar dicionário de chave secreta
    chave_secreta = {}
    blocos_formulario_txt = []
    registros_cegados_csv = []
    
    label_alias = ['Modelo A', 'Modelo B', 'Modelo C']
    
    blocos_formulario_txt.append("=" * 70)
    blocos_formulario_txt.append("  FORMULÁRIO DE AVALIAÇÃO HUMANA — PRONTO PARA COPIAR E COLAR")
    blocos_formulario_txt.append("=" * 70)
    blocos_formulario_txt.append("Instruções: Para cada bloco abaixo, crie uma pergunta no Google Forms.")
    blocos_formulario_txt.append("Copie o [PROMPT DO USUÁRIO] e as opções de [RESPOSTAS ANONIMIZADAS].\n")
    
    for i, prompt_id in enumerate(prompts_selecionados, 1):
        df_prompt = df_selecionado[df_selecionado['prompt_id'] == prompt_id].copy()
        
        categoria = df_prompt['categoria'].iloc[0]
        mensagem_prompt = df_prompt['mensagem'].iloc[0]
        
        # Embaralhar a ordem das IAs para este prompt específico
        provedores = list(df_prompt['provedor'].unique())
        random.shuffle(provedores)
        
        # Mapeamento do cegamento para este prompt
        mapeamento_prompt = {}
        for alias, prov in zip(label_alias, provedores):
            mapeamento_prompt[alias] = prov
            
        chave_secreta[prompt_id] = mapeamento_prompt
        
        # Montar texto do bloco para o Forms
        bloco_str = f"--- BLOCO {i:02d}/20 | Categoria: {categoria} (ID: {prompt_id}) ---\n\n"
        bloco_str += f"💬 MENSAGEM DO USUÁRIO:\n\"{mensagem_prompt}\"\n\n"
        bloco_str += "🤖 RESPOSTAS PARA AVALIAR:\n\n"
        
        for alias in label_alias:
            prov_real = mapeamento_prompt[alias]
            resp_row = df_prompt[df_prompt['provedor'] == prov_real].iloc[0]
            texto_resp = resp_row['resposta']
            
            bloco_str += f"[{alias}]:\n\"{texto_resp}\"\n\n"
            
            # Guardar linha para o CSV cegado
            registros_cegados_csv.append({
                "bloco_num": i,
                "prompt_id": prompt_id,
                "categoria": categoria,
                "mensagem": mensagem_prompt,
                "modelo_cegado": alias,
                "provedor_real": prov_real,
                "resposta": texto_resp
            })
            
        bloco_str += "-" * 70 + "\n"
        blocos_formulario_txt.append(bloco_str)
        
    # 4. Salvar Chave Secreta JSON
    OUTPUT_KEY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_KEY_JSON, "w", encoding="utf-8") as f:
        json.dump(chave_secreta, f, ensure_ascii=False, indent=2)
        
    # 5. Salvar Arquivo de Texto do Formulário
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(blocos_formulario_txt))
        
    # 6. Salvar CSV Cegado
    df_cegado_csv = pd.DataFrame(registros_cegados_csv)
    df_cegado_csv.to_csv(OUTPUT_CSV_BLIND, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 65)
    print("SUCESSO! ARQUIVOS GERADOS COM ÊXITO:")
    print("=" * 65)
    print(f"1. Arquivo de Texto para Copiar/Colar no Forms:\n   -> {OUTPUT_TXT.resolve()}")
    print(f"2. Chave Secreta de Cegamento (JSON):\n   -> {OUTPUT_KEY_JSON.resolve()}")
    print(f"3. Dataset Cegado em CSV:\n   -> {OUTPUT_CSV_BLIND.resolve()}")

if __name__ == "__main__":
    gerar_formulario_humanos()
