import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Baixar stop words em português
nltk.download('stopwords')
stop_words_pt = stopwords.words('portuguese')

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíóòôõúç\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def categorizar_rating(rating):
    """Converte avaliação de 1 a 5 estrelas em rótulos de sentimento."""
    if rating <= 2:
        return 'NEGATIVO'
    elif rating == 3:
        return 'NEUTRO'
    else:
        return 'POSITIVO'

print("=" * 65)
print("1. CARREGANDO DATASET DE LINGUAGEM NATURAL (B2W-REVIEWS PT-BR)")
print("=" * 65)

local_b2w = 'data/input/b2w_reviews_sample.csv'
if not os.path.exists(local_b2w):
    print("Baixando dataset oficial de avaliacoes em Portugues do Brasil...")
    url = 'https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv'
    df_raw = pd.read_csv(url, low_memory=False).dropna(subset=['review_text', 'overall_rating'])
    df_raw = df_raw.sample(n=10000, random_state=42)
    os.makedirs('data/input', exist_ok=True)
    df_raw[['review_text', 'overall_rating']].to_csv(local_b2w, index=False, encoding='utf-8-sig')
    print("Dataset salvo localmente em 'data/input/b2w_reviews_sample.csv'!")

df_b2w = pd.read_csv(local_b2w)
print(f"Total de frases limpas para treino: {len(df_b2w)}")

# 2. Rotular e Limpar o Texto
df_b2w['label'] = df_b2w['overall_rating'].apply(categorizar_rating)
df_b2w['text_clean'] = df_b2w['review_text'].apply(limpar_texto)

# 3. Vetorização TF-IDF
print("\n2. Extraindo Features com TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words=stop_words_pt,
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(df_b2w['text_clean'])
y = df_b2w['label']

# 4. Divisão 80% Treino e 20% Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Frases para Treino: {X_treino.shape[0]} | Frases para Teste: {X_teste.shape[0]}")

# 5. Instanciar e Treinar a Random Forest
print("\n3. Treinando a Random Forest (150 arvores, classe balanceada)...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_treino, y_treino)
print("[OK] Treino concluido com sucesso!")

# 6. Boletim de Desempenho
y_pred = rf_model.predict(X_teste)
print("\n" + "=" * 65)
print("BOLETIM DE DESEMPENHO DA RANDOM FOREST (DATASET B2W PT-BR)")
print("=" * 65)
print(f"Acuracia Geral no Teste: {accuracy_score(y_teste, y_pred) * 100:.2f}%\n")
print(classification_report(y_teste, y_pred))

# 7. Palavras Mais Decisivas do Modelo
importances = rf_model.feature_importances_
features = vectorizer.get_feature_names_out()
top_indices = importances.argsort()[-10:][::-1]
print("\nTop 10 Palavras Mais Decisivas na Random Forest:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. {features[idx]} (peso: {importances[idx]:.4f})")

# 8. Classificar as 300 Respostas de IA do seu TG
csv_ias_path = 'data/output/respostas_ias_tg.csv'
if os.path.exists(csv_ias_path):
    print("\n" + "=" * 65)
    print(f"4. CLASSIFICANDO AS 300 RESPOSTAS DE IA ({csv_ias_path})")
    print("=" * 65)
    
    df_tg = pd.read_csv(csv_ias_path)
    df_tg['resposta_limpa'] = df_tg['resposta'].apply(limpar_texto)
    
    X_tg = vectorizer.transform(df_tg['resposta_limpa'])
    
    df_tg['rf_label'] = rf_model.predict(X_tg)
    df_tg['rf_score'] = rf_model.predict_proba(X_tg).max(axis=1).round(4)
    
    df_tg.drop(columns=['resposta_limpa'], inplace=True, errors='ignore')
    out_csv = 'data/output/respostas_analisadas_rf.csv'
    df_tg.to_csv(out_csv, index=False, encoding='utf-8-sig')
    
    print(f"[SUCESSO] 300 respostas classificadas e salvas em:\n-> {out_csv}")
    print("\nDistribuição das Predições da Random Forest:")
    print(df_tg['rf_label'].value_counts())
