import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_CSV = Path(os.getenv("INPUT_CSV", "data/input/prompts_vulnerabilidade.csv"))
OUTPUT_CSV = Path(os.getenv("OUTPUT_CSV", "data/output/respostas_ias.csv"))

MODELS = {
    "OpenAI": os.getenv("OPENAI_MODEL", "openai/gpt-5.4"),
    "Google": os.getenv("GEMINI_MODEL", "google/gemini-3.6-flash"),
    "Anthropic": os.getenv("CLAUDE_MODEL", "anthropic/claude-sonnet-4.6"),
}

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DELAY_BETWEEN_CALLS = float(os.getenv("DELAY_BETWEEN_CALLS", "1.0"))

OUTPUT_FIELDS = [
    "response_id", "prompt_id", "categoria", "intensidade", "mensagem",
    "provedor", "modelo", "resposta", "prompt_tokens", "completion_tokens",
    "total_tokens", "tempo_segundos", "status", "erro", "coletado_em_utc",
]

def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variável {name} não encontrada. Crie o .env a partir de .env.example.")
    return value

def make_client():
    return OpenAI(api_key=require_env("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

def load_prompts(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("O dataset está vazio.")
    required = {"prompt_id", "categoria", "intensidade", "mensagem"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    return rows

def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK":
                completed.add((row.get("prompt_id"), row.get("modelo")))
    return completed

def append_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(result)

def call_model(client, model, prompt):
    # Sem system prompt adicional: todos recebem exatamente a mensagem do dataset.
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    pt = getattr(usage, "prompt_tokens", "") if usage else ""
    ct = getattr(usage, "completion_tokens", "") if usage else ""
    tt = getattr(usage, "total_tokens", "") if usage else ""
    return text.strip(), pt, ct, tt

def run_with_retry(client, model, prompt):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text, pt, ct, tt = call_model(client, model, prompt)
            return text, pt, ct, tt, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                wait = min(2 ** (attempt - 1), 8)
                print(f"    Erro: {last_error}")
                print(f"    Nova tentativa em {wait}s...")
                time.sleep(wait)
    return "", "", "", "", last_error

def main():
    parser = argparse.ArgumentParser(description="Coleta respostas para o TG via OpenRouter.")
    parser.add_argument("--limit", type=int, default=None, help="Processa apenas os primeiros N prompts.")
    parser.add_argument("--provider", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--force", action="store_true", help="Refaz respostas já coletadas.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit deve ser maior que zero.")

    client = make_client()
    prompts = load_prompts(INPUT_CSV)
    if args.limit is not None:
        prompts = prompts[:args.limit]

    selected = [args.provider] if args.provider else list(MODELS.keys())
    completed = set() if args.force else load_completed(OUTPUT_CSV)
    total = len(prompts) * len(selected)
    current = 0

    print("=" * 64)
    print("COLETOR DE RESPOSTAS — TG")
    print("=" * 64)
    print(f"Dataset: {INPUT_CSV.resolve()}")
    print(f"Saída: {OUTPUT_CSV.resolve()}")
    print(f"Prompts: {len(prompts)} | Modelos: {len(selected)} | Chamadas: {total}")
    print(f"Máximo de tokens de saída: {MAX_TOKENS}")
    print("=" * 64)

    for prompt_row in prompts:
        prompt_id = prompt_row["prompt_id"]
        mensagem = prompt_row["mensagem"]
        for provider in selected:
            current += 1
            model = MODELS[provider]
            if (prompt_id, model) in completed:
                print(f"[{current}/{total}] {prompt_id} | {provider} | já coletado")
                continue

            print(f"[{current}/{total}] {prompt_id} | {provider} | {model}")
            start = time.perf_counter()
            text, pt, ct, tt, error = run_with_retry(client, model, mensagem)
            elapsed = round(time.perf_counter() - start, 3)
            result = {
                "response_id": f"{prompt_id}_{provider.upper()}",
                "prompt_id": prompt_id,
                "categoria": prompt_row["categoria"],
                "intensidade": prompt_row["intensidade"],
                "mensagem": mensagem,
                "provedor": provider,
                "modelo": model,
                "resposta": text,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "tempo_segundos": elapsed,
                "status": "OK" if error is None else "ERRO",
                "erro": error or "",
                "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_result(OUTPUT_CSV, result)
            print(f"    {'OK' if error is None else 'FALHOU'} | {elapsed}s | tokens: {tt if tt != '' else 'n/d'}")
            if error:
                print(f"    {error}")
            time.sleep(DELAY_BETWEEN_CALLS)

    print("=" * 64)
    print(f"COLETA ENCERRADA — {OUTPUT_CSV.resolve()}")
    print("=" * 64)

if __name__ == "__main__":
    main()
