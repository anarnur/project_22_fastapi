from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from app.models import PredictionRequest, PredictionResponse
from app.inference import inference_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загрузка модели при старте
    try:
        inference_service.load_model()
        yield
    finally:
        pass

app = FastAPI(
    title="AI Production API",
    description="API для генерации текста с помощью дообученной модели",
    lifespan=lifespan
)

# --- ДОБАВЛЯЕМ CORS (Разрешаем запросы от фронтенда) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает запросы со всех адресов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "AI Production API",
        "status": "active",
        "model": inference_service.model_path
    }

# Обычный эндпоинт (выдает весь текст сразу)
@app.post("/generate", response_model=PredictionResponse)
async def generate_text(request: PredictionRequest):
    try:
        # Важно: убедись, что метод generate() есть в твоем inference.py
        result = inference_service.generate(request.prompt)
        return PredictionResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ДОБАВЛЯЕМ СТРИМИНГ (для Проекта 25) ---
@app.post("/generate_stream")
async def generate_stream(request: PredictionRequest):
    try:
        # Используем генератор из ModelInference
        return StreamingResponse(
            inference_service.generate_stream(request.prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    