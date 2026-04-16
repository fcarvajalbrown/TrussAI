from strands.models.ollama import OllamaModel

# swap model_id here if you change models later
model = OllamaModel(
    model_id="ministral-3:3b",
    host="http://localhost:11434",
)