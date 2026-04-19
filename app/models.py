from pydantic import BaseModel, Field, field_validator

# Схема для входящего запроса
class PredictionRequest(BaseModel):
    # Поле prompt обязательно, тип — строка [cite: 46]
    prompt: str = Field(
        ..., 
        description="Текстовый промпт для модели",
        max_length=2048 # Ограничение длины [cite: 47]
    )

    # Валидация: промпт не должен быть пустым [cite: 47]
    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Промпт не может быть пустым")
        return v

# Схема для исходящего ответа
class PredictionResponse(BaseModel):
    # Поле с результатом генерации [cite: 46]
    generated_text: str