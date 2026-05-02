import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import os
from dotenv import load_dotenv

load_dotenv()

class ModelInference:
    def __init__(self):
        # Используем TinyLlama, так как теперь лимиты позволяют (Hobby план)
        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Загрузка модели при старте приложения"""
        print(f"Начинаю загрузку модели: {self.model_path}", flush=True) 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32, 
                device_map="cpu",
                trust_remote_code=True
            )
            print(f"Модель {self.model_path} успешно загружена!", flush=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}", flush=True)

    def generate_stream(self, prompt: str):
        """Логика потоковой генерации (Streaming) для Проекта 25"""
        # Форматируем промпт специально для TinyLlama Chat
        formatted_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Запускаем генерацию в отдельном потоке, чтобы не блокировать итератор
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

# Создаем экземпляр
inference_service = ModelInference()