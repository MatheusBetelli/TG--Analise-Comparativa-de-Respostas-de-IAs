from transformers import pipeline

MODEL_NAME = "Adilmar/caramelo-smile"

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME
)

texto = "Estou muito feliz com o resultado que consegui."

resultado = classifier(texto)

print(resultado)