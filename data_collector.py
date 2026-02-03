import os
import json
import pandas as pd
import time
import logging
import requests
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime
from pathlib import Path

# Импорты из проекта
from prompts import PROMPTS
from config import (
    BOTHUB_API_KEY, BOTHUB_BASE_URL,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    MAX_RETRIES, REQUEST_DELAY, MAX_TOKENS, TEMPERATURE,
    CHECKPOINT_INTERVAL, LOG_FILE, LOG_LEVEL
)


def telegram_log(message: str):
    """Отправка логов в Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        logging.warning(f"Telegram logging error: {e}")


class ResponseCollector:
    """Сборщик ответов моделей"""
    
    def __init__(self, json_dir: str, output_dir: str = "raw_responses"):
        self.json_dir = json_dir
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Инициализация OpenAI клиента для Bothub
        self.client = OpenAI(
            base_url=BOTHUB_BASE_URL,
            api_key=BOTHUB_API_KEY
        )
        
        # Настройка логирования
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.responses = []
        self.failed_requests = []
        self.start_time = None
        
    def load_json_tables(self) -> List[Dict[str, Any]]:
        """Загрузка JSON таблиц из директории"""
        tables = []
        json_files = list(Path(self.json_dir).glob("*.json"))
        
        if not json_files:
            logging.warning(f"Не найдено JSON файлов в директории: {self.json_dir}")
            return tables
        
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Поддержка формата с data и headers
                    if isinstance(data, dict) and "data" in data:
                        table = data.get("data", [])
                        true_coords = data.get("headers", [])
                        if not table:
                            logging.warning(f"Пустая таблица в {filepath.name}, пропускаем")
                            continue
                    # Поддержка формата массива таблиц (PMC формат)
                    elif isinstance(data, list) and len(data) > 0:
                        # Если это массив таблиц, обрабатываем каждую
                        for idx, table_obj in enumerate(data):
                            if isinstance(table_obj, dict) and "data" in table_obj:
                                table = table_obj.get("data", [])
                                true_coords = table_obj.get("headers", [])
                                if not table:
                                    continue
                                tables.append({
                                    "file": f"{filepath.stem}_table_{idx}.json",
                                    "table_data": table,
                                    "true_coords": [(h['row'], h['col']) for h in true_coords] if true_coords else []
                                })
                        continue
                    else:
                        logging.warning(f"Неизвестный формат данных в {filepath.name}, пропускаем")
                        continue
                    
                    # Валидация координат заголовков
                    true_coords_valid = []
                    for h in true_coords:
                        if isinstance(h, dict) and 'row' in h and 'col' in h:
                            try:
                                row = int(h['row'])
                                col = int(h['col'])
                                if row >= 0 and col >= 0:
                                    true_coords_valid.append((row, col))
                            except (ValueError, TypeError):
                                logging.warning(f"Некорректные координаты заголовка в {filepath.name}: {h}")
                    
                    tables.append({
                        "file": filepath.name,
                        "table_data": table,
                        "true_coords": true_coords_valid
                    })
            except json.JSONDecodeError as e:
                logging.error(f"Ошибка парсинга JSON в {filepath.name}: {e}")
            except Exception as e:
                logging.error(f"Ошибка чтения {filepath.name}: {e}")
        
        logging.info(f"Загружено {len(tables)} JSON-таблиц")
        return tables
    
    def table_to_text(self, table_data: List[List[Any]]) -> str:
        """Конвертация таблицы в markdown формат"""
        if not table_data or len(table_data) == 0:
            return ""
        
        # Безопасная конвертация в строки
        def safe_str(val):
            if val is None:
                return ""
            return str(val).strip()
        
        rows = []
        for row in table_data:
            row_str = "| " + " | ".join(safe_str(cell) for cell in row) + " |"
            rows.append(row_str)
        
        return "\n".join(rows)
    
    def create_strict_system_prompt(self) -> str:
        """Жесткий системный промпт для контроля формата ответа"""
        return """You are a precise table header detection system.

CRITICAL RULES - MUST FOLLOW:
1. You MUST respond ONLY with valid JSON
2. NO explanations, NO reasoning, NO markdown, NO additional text
3. Use EXACTLY this format: {"headers": [{"row": 0, "col": 0}]}
4. "row" and "col" must be non-negative integers (0-indexed)
5. Include ALL and ONLY cells that are table headers
6. Empty headers list is valid: {"headers": []}

RESPONSE FORMAT:
{"headers": [{"row": <int>, "col": <int>}]}

VALID EXAMPLES:
{"headers": [{"row": 0, "col": 0}, {"row": 0, "col": 1}, {"row": 0, "col": 2}]}
{"headers": [{"row": 0, "col": 0}, {"row": 1, "col": 0}]}
{"headers": []}

INVALID EXAMPLES (DO NOT DO THIS):
````json {"headers": [...]} ```  ← NO markdown
Here are the headers: {"headers": [...]}  ← NO extra text
{"headers": [{"row": "0", "col": "1"}]}  ← row/col must be integers

RESPOND WITH ONLY THE JSON OBJECT."""

    def prepare_messages(self, prompt_config: Dict, table_text: str) -> List[Dict]:
        """Подготовка сообщений для API из конфига промпта"""
        messages = []
        
        # Строгий системный промпт + промпт из конфига
        strict_prompt = self.create_strict_system_prompt()
        
        if prompt_config.get("system"):
            # Объединяем промпты
            config_system = "\n".join(prompt_config["system"])
            combined_system = f"{strict_prompt}\n\n--- Additional Instructions ---\n{config_system}"
            messages.append({"role": "system", "content": combined_system})
        else:
            messages.append({"role": "system", "content": strict_prompt})
        
        # User prompt с подстановкой таблицы
        user_prompt = prompt_config["user"].format(table_text=table_text)
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def make_api_call(self, model: str, messages: List[Dict], max_retries: int = MAX_RETRIES) -> Dict:
        """API вызов с retry логикой и обработкой ошибок"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Пытаемся использовать response_format если поддерживается
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        response_format={"type": "json_object"}
                    )
                except Exception:
                    # Fallback без response_format
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )
                
                duration = time.time() - start_time
                response_text = completion.choices[0].message.content
                
                # Извлечение информации о токенах
                tokens_info = None
                if hasattr(completion, 'usage') and completion.usage:
                    tokens_info = {
                        "prompt": completion.usage.prompt_tokens,
                        "completion": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens
                    }
                
                return {
                    "success": True,
                    "response": response_text,
                    "duration": duration,
                    "tokens_used": tokens_info,
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model}: {error_msg}")
                
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": error_msg,
                        "attempt": attempt + 1
                    }
                
                # Exponential backoff
                sleep_time = min(2 ** attempt, 30)
                time.sleep(sleep_time)
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def collect_responses(self, models: List[str], prompts: List[Dict[str, Any]]):
        """Основной цикл сбора ответов"""
        tables = self.load_json_tables()
        
        if not tables:
            logging.error("Нет таблиц для обработки!")
            return
        
        total_tasks = len(models) * len(prompts) * len(tables)
        self.start_time = datetime.now()
        
        # Telegram уведомление о старте
        telegram_log(
            f"🚀 СТАРТ ЭКСПЕРИМЕНТА\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Моделей: {len(models)}\n"
            f"📝 Промптов: {len(prompts)}\n"
            f"📄 Таблиц: {len(tables)}\n"
            f"🎯 Всего запросов: {total_tasks}\n"
            f"⏰ Начало: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        logging.info(
            f"Starting collection: {len(models)} models × {len(prompts)} prompts × {len(tables)} tables = {total_tasks} requests"
        )
        
        processed = 0
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        for model_idx, model in enumerate(models):
            logging.info(f"\n{'='*70}\nМодель [{model_idx+1}/{len(models)}]: {model}\n{'='*70}")
            
            for prompt_idx, prompt_config in enumerate(prompts):
                prompt_name = prompt_config.get('name', f'prompt_{prompt_idx}')
                logging.info(f"  Промпт [{prompt_idx+1}/{len(prompts)}]: {prompt_name}")
                
                for table_idx, tbl in enumerate(tables):
                    
                    # Формируем сообщения
                    table_text = self.table_to_text(tbl['table_data'])
                    messages = self.prepare_messages(prompt_config, table_text)
                    
                    # API вызов
                    result = self.make_api_call(model, messages)
                    
                    # Подготовка данных для сохранения
                    # Безопасное получение размеров таблицы
                    table_rows = len(tbl['table_data']) if tbl['table_data'] else 0
                    table_cols = len(tbl['table_data'][0]) if tbl['table_data'] and len(tbl['table_data']) > 0 else 0
                    
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "model_idx": model_idx,
                        "prompt_name": prompt_name,
                        "prompt_idx": prompt_idx,
                        "table_file": tbl['file'],
                        "table_idx": table_idx,
                        "table_rows": table_rows,
                        "table_cols": table_cols,
                        "true_headers": tbl['true_coords'],
                        "true_headers_count": len(tbl['true_coords']),
                        "api_success": result["success"],
                        "raw_response": result.get("response", ""),
                        "error_message": result.get("error", ""),
                        "duration_sec": result.get("duration", 0),
                        "tokens": result.get("tokens_used"),
                        "retry_attempts": result.get("attempt", 1),
                    }
                    
                    # Сохраняем в соответствующий список
                    if result["success"]:
                        self.responses.append(response_data)
                    else:
                        self.failed_requests.append(response_data)
                        logging.error(
                            f"❌ Failed: {model} | {prompt_name} | {tbl['file']} | Error: {result.get('error', 'Unknown')}"
                        )
                    
                    processed += 1
                    
                    # Периодическое сохранение и отчет
                    if processed % CHECKPOINT_INTERVAL == 0 or processed == total_tasks:
                        self.save_checkpoint(timestamp)
                        self._log_progress(processed, total_tasks)
                    
                    # Задержка между запросами
                    if processed < total_tasks:
                        time.sleep(REQUEST_DELAY)
        
        # Финальное сохранение
        self.save_final_results(timestamp)
        self._log_completion(total_tasks)
    
    def _log_progress(self, processed: int, total: int):
        """Логирование прогресса"""
        success_count = len(self.responses)
        failed_count = len(self.failed_requests)
        success_rate = (success_count / processed * 100) if processed > 0 else 0
        
        elapsed = datetime.now() - self.start_time
        avg_time = elapsed.total_seconds() / processed if processed > 0 else 0
        eta_seconds = avg_time * (total - processed)
        eta = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
        
        progress_pct = (processed / total * 100) if total > 0 else 0.0
        message = (
            f"📊 ПРОГРЕСС\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Обработано: {processed}/{total} ({progress_pct:.1f}%)\n"
            f"🎯 Успешных: {success_count}\n"
            f"❌ Ошибок: {failed_count}\n"
            f"📈 Success Rate: {success_rate:.1f}%\n"
            f"⏱ Среднее время: {avg_time:.2f}s\n"
            f"⏰ ETA: {eta}"
        )
        
        telegram_log(message)
        logging.info(f"Progress: {processed}/{total} | Success: {success_rate:.1f}% | ETA: {eta}")
    
    def _log_completion(self, total: int):
        """Логирование завершения"""
        elapsed = datetime.now() - self.start_time
        success_count = len(self.responses)
        failed_count = len(self.failed_requests)
        
        success_rate_final = (success_count / total * 100) if total > 0 else 0.0
        message = (
            f"✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Всего запросов: {total}\n"
            f"✅ Успешных: {success_count}\n"
            f"❌ Ошибок: {failed_count}\n"
            f"📈 Success Rate: {success_rate_final:.1f}%\n"
            f"⏱ Общее время: {elapsed}\n"
            f"📁 Результаты в: {self.output_dir}"
        )
        
        telegram_log(message)
        logging.info(f"\n{'='*70}\n{message}\n{'='*70}")
    
    def save_checkpoint(self, timestamp: str):
        """Промежуточное сохранение"""
        checkpoint_file = Path(self.output_dir) / f"checkpoint_{timestamp}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "responses": self.responses,
                "failed": self.failed_requests,
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "total_collected": len(self.responses),
                    "total_failed": len(self.failed_requests),
                    "elapsed_time": str(datetime.now() - self.start_time)
                }
            }, f, ensure_ascii=False, indent=2)
        
        logging.debug(f"Checkpoint saved: {checkpoint_file}")
    
    def save_final_results(self, timestamp: str):
        """Финальное сохранение результатов"""
        
        # 1. Полный JSON файл
        json_file = Path(self.output_dir) / f"responses_{timestamp}.json"
        total_time = datetime.now() - self.start_time
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "collection_date": timestamp,
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_duration": str(total_time),
                    "total_collected": len(self.responses),
                    "total_failed": len(self.failed_requests),
                    "success_rate": len(self.responses) / (len(self.responses) + len(self.failed_requests)) * 100 if (len(self.responses) + len(self.failed_requests)) > 0 else 0,
                    "unique_models": len(set(r['model'] for r in self.responses)),
                    "unique_prompts": len(set(r['prompt_name'] for r in self.responses)),
                    "unique_tables": len(set(r['table_file'] for r in self.responses)),
                },
                "responses": self.responses,
                "failed": self.failed_requests,
            }, f, ensure_ascii=False, indent=2)
        
        # 2. CSV для быстрого просмотра
        if self.responses:
            df = pd.DataFrame(self.responses)
            csv_file = Path(self.output_dir) / f"responses_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logging.info(f"CSV saved: {csv_file}")
        
        # 3. Отдельный файл с ошибками
        if self.failed_requests:
            failed_file = Path(self.output_dir) / f"failed_{timestamp}.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_failed": len(self.failed_requests),
                    "failed_requests": self.failed_requests
                }, f, ensure_ascii=False, indent=2)
            logging.warning(f"Failed requests saved: {failed_file}")
        
        # 4. Краткая сводка
        summary_file = Path(self.output_dir) / f"summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Duration: {total_time}\n")
            f.write(f"Total Requests: {len(self.responses) + len(self.failed_requests)}\n")
            f.write(f"Successful: {len(self.responses)}\n")
            f.write(f"Failed: {len(self.failed_requests)}\n")
            f.write(f"Success Rate: {len(self.responses)/(len(self.responses)+len(self.failed_requests))*100:.2f}%\n\n")
            
            if self.responses:
                df = pd.DataFrame(self.responses)
                f.write("Average Duration per Request: {:.2f}s\n".format(df['duration_sec'].mean()))
                if df['tokens'].notna().any():
                    total_tokens = sum(t['total'] for t in df['tokens'].dropna() if t and 'total' in t)
                    f.write(f"Total Tokens Used: {total_tokens}\n")
        
        logging.info(f"Results saved to {self.output_dir}")
        print(f"\n✅ ФАЙЛЫ СОХРАНЕНЫ:")
        print(f"   📄 {json_file}")
        if self.responses:
            print(f"   📊 {csv_file}")
        if self.failed_requests:
            print(f"   ❌ {failed_file}")
        print(f"   📝 {summary_file}")


if __name__ == "__main__":
    # Простой тест
    from config import JSON_DIR, get_models
    
    test_models = get_models()[:2]  # Берем первые 2 модели
    test_prompts = PROMPTS[:3]  # Берем первые 3 промпта
    
    print(f"Test run with {len(test_models)} models and {len(test_prompts)} prompts")
    
    collector = ResponseCollector(
        json_dir=JSON_DIR,
        output_dir="test_responses"
    )
    
    collector.collect_responses(test_models, test_prompts)

