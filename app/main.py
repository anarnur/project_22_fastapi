from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import PredictionRequest, PredictionResponse
from app.inference import inference_service

# Используем lifespan для загрузки модели один раз при старте [cite: 94]
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Логика при запуске [cite: 53]
    try:
        inference_service.load_model()
        yield
    finally:
        # Логика при остановке (если нужно очистить память)
        pass

app = FastAPI(
    title="AI Production API",
    description="API для генерации текста с помощью дообученной модели",
    lifespan=lifespan
)

# Эндпоинт для проверки работоспособности [cite: 17, 56]
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_path": inference_service.model_path
    }

# Эндпоинт для генерации текста [cite: 55, 57]
@app.post("/generate", response_model=PredictionResponse)
async def generate_text(request: PredictionRequest):
    try:
        # Вызываем функцию генерации [cite: 60]
        result = inference_service.generate(request.prompt)
        return PredictionResponse(generated_text=result)
    except Exception as e:
        # Обработка внутренних ошибок [cite: 62, 75]
        raise HTTPException(status_code=500, detail=str(e))