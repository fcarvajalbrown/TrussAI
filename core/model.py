from strands.models.ollama import OllamaModel

# swap model_id here if you change models later
model = OllamaModel(
    model_id="qwen3.5:2b",
    host="http://localhost:11434",
)