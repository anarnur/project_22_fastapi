import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv() # Загружаем переменные из .env

class ModelInference:
    def __init__(self):
        # Прямо прописываем путь, игнорируя всё остальное
        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Загрузка модели и токенизатора один раз при старте"""
        print(f"Начинаю загрузку модели: {self.model_path}") 
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Загружаем саму модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32, # Оптимально для процессоров Railway
            device_map="cpu",          # Указываем явно, так как на бесплатном тарифе нет GPU
            trust_remote_code=True
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