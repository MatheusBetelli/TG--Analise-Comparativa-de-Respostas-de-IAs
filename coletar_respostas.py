import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

INPUT_CSV = Path(
    os.getenv(
        "INPUT_CSV",
        "data/input/mensagens_tg.csv"
    )
)

OUTPUT_CSV = Path(
    os.getenv(
        "OUTPUT_CSV",
        "data/output/respostas_ias_tg.csv"
    )
)

MODELS = {
    "OpenAI": os.getenv(
        "OPENAI_MODEL",
        "openai/gpt-5.4"
    ),
    "Anthropic": os.getenv(
        "CLAUDE_MODEL",
        "anthropic/claude-sonnet-4.6"
    ),
    "Meta": os.getenv(
        "META_MODEL",
        "meta-llama/llama-4-maverick"
    ),
}

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "600")
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "3")
)

DELAY_BETWEEN_CALLS = float(
    os.getenv("DELAY_BETWEEN_CALLS", "1.0")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.0")
)

OUTPUT_FIELDS = [
    "response_id",
    "mensagem_id",
    "categoria",
    "mensagem_original",
    "mensagem_pt",
    "provedor",
    "modelo",
    "resposta",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tempo_segundos",
    "status",
    "erro",
    "coletado_em_utc",
]


def require_env(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Variável {name} não encontrada no arquivo .env."
        )

    return value


def make_client():
    return OpenAI(
        api_key=require_env("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )


def load_messages(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        rows = list(csv.DictReader(arquivo))

    if not rows:
        raise ValueError("O dataset está vazio.")

    required = {
        "mensagem_id",
        "categoria",
        "mensagem_original",
        "mensagem_pt",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {sorted(missing)}"
        )

    for row in rows:
        if not row["mensagem_pt"].strip():
            raise ValueError(
                f"A mensagem {row['mensagem_id']} ainda não foi traduzida."
            )

    return rows


def load_completed(path):
    completed = set()

    if not path.exists():
        return completed

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        for row in csv.DictReader(arquivo):
            if row.get("status") == "OK":
                completed.add(
                    (
                        row.get("mensagem_id"),
                        row.get("modelo")
                    )
                )

    return completed


def append_result(path, result):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=OUTPUT_FIELDS
        )

        if not exists:
            writer.writeheader()

        writer.writerow(result)


def call_model(client, model, mensagem):
    # Nenhum system prompt adicional.
    # Todos os modelos recebem exatamente a mesma mensagem traduzida.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": mensagem
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    text = (
        response.choices[0].message.content
        or ""
    ).strip()

    usage = getattr(
        response,
        "usage",
        None
    )

    prompt_tokens = (
        getattr(usage, "prompt_tokens", "")
        if usage
        else ""
    )

    completion_tokens = (
        getattr(usage, "completion_tokens", "")
        if usage
        else ""
    )

    total_tokens = (
        getattr(usage, "total_tokens", "")
        if usage
        else ""
    )

    return (
        text,
        prompt_tokens,
        completion_tokens,
        total_tokens
    )


def run_with_retry(client, model, mensagem):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            return (
                *call_model(
                    client,
                    model,
                    mensagem
                ),
                None
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < MAX_RETRIES:
                wait = min(
                    2 ** (attempt - 1),
                    8
                )

                print(
                    f"    Erro: {last_error}"
                )

                print(
                    f"    Nova tentativa em {wait}s..."
                )

                time.sleep(wait)

    return "", "", "", "", last_error


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Coleta respostas dos três LLMs "
            "para o TG via OpenRouter."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa apenas as primeiras N mensagens."
    )

    parser.add_argument(
        "--provider",
        choices=list(MODELS.keys()),
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz respostas já coletadas."
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise ValueError(
            "--limit deve ser maior que zero."
        )

    client = make_client()
    mensagens = load_messages(INPUT_CSV)

    if args.limit is not None:
        mensagens = mensagens[:args.limit]

    selected = (
        [args.provider]
        if args.provider
        else list(MODELS.keys())
    )

    completed = (
        set()
        if args.force
        else load_completed(OUTPUT_CSV)
    )

    total = (
        len(mensagens)
        * len(selected)
    )

    current = 0

    print("=" * 64)
    print("COLETOR DE RESPOSTAS — TG")
    print("=" * 64)
    print(f"Dataset: {INPUT_CSV.resolve()}")
    print(f"Saída: {OUTPUT_CSV.resolve()}")
    print(
        f"Mensagens: {len(mensagens)} | "
        f"Modelos: {len(selected)} | "
        f"Chamadas: {total}"
    )
    print(
        f"Máximo de tokens: {MAX_TOKENS} | "
        f"Temperatura: {TEMPERATURE}"
    )
    print("=" * 64)

    for row in mensagens:
        mensagem_id = row["mensagem_id"]
        mensagem_pt = row["mensagem_pt"]

        for provider in selected:
            current += 1
            model = MODELS[provider]

            if (
                mensagem_id,
                model
            ) in completed:
                print(
                    f"[{current}/{total}] "
                    f"{mensagem_id} | "
                    f"{provider} | já coletado"
                )
                continue

            print(
                f"[{current}/{total}] "
                f"{mensagem_id} | "
                f"{provider} | {model}"
            )

            start = time.perf_counter()

            (
                text,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                error
            ) = run_with_retry(
                client,
                model,
                mensagem_pt
            )

            elapsed = round(
                time.perf_counter() - start,
                3
            )

            result = {
                "response_id":
                    f"{mensagem_id}_{provider.upper()}",

                "mensagem_id":
                    mensagem_id,

                "categoria":
                    row["categoria"],

                "mensagem_original":
                    row["mensagem_original"],

                "mensagem_pt":
                    mensagem_pt,

                "provedor":
                    provider,

                "modelo":
                    model,

                "resposta":
                    text,

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    total_tokens,

                "tempo_segundos":
                    elapsed,

                "status":
                    "OK"
                    if error is None
                    else "ERRO",

                "erro":
                    error or "",

                "coletado_em_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
            }

            append_result(
                OUTPUT_CSV,
                result
            )

            print(
                f"    "
                f"{'OK' if error is None else 'FALHOU'} "
                f"| {elapsed}s "
                f"| tokens: "
                f"{total_tokens if total_tokens != '' else 'n/d'}"
            )

            if error:
                print(f"    {error}")

            time.sleep(
                DELAY_BETWEEN_CALLS
            )

    print("=" * 64)
    print(
        f"COLETA ENCERRADA — "
        f"{OUTPUT_CSV.resolve()}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
