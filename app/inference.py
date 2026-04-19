import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv() # Загружаем переменные из .env

class ModelInference:
    def __init__(self):
        # Берем путь из .env или используем заглушку gpt2 для теста
        self.model_path = os.getenv("MODEL_PATH", "gpt2") 
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Загрузка модели и токенизатора один раз при старте"""
        print(f"Загрузка модели из: {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("Модель успешно загружена!")

    def generate(self, prompt: str) -> str:
        """Логика генерации текста"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Создаем один экземпляр класса для всего приложения
inference_service = ModelInference()