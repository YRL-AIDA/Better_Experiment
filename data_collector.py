import os
import json
import pandas as pd
import time
import logging
import requests
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Секреты загружаются из .env (файл в .gitignore)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def telegram_log(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        print(f"Telegram logging error: {e}")


class ResponseCollector:
    """Только сбор ответов моделей без анализа"""
    
    def __init__(self, json_dir: str, output_dir: str = "raw_responses"):
        self.json_dir = json_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        api_key = os.getenv("BOTHUB_API_KEY", "")
        if not api_key:
            raise ValueError(
                "BOTHUB_API_KEY не задан. Создайте файл .env и укажите BOTHUB_API_KEY=your_key"
            )
        self.client = OpenAI(
            base_url="https://api.bothub.chat/v1",
            api_key=api_key
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('collection.log'),
                logging.StreamHandler()
            ]
        )
        
        self.responses = []
        self.failed_requests = []
        
    def load_json_tables(self) -> List[Dict[str, Any]]:
        tables = []
        for filename in os.listdir(self.json_dir):
            if filename.endswith(".json"):
                full_path = os.path.join(self.json_dir, filename)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        table = data.get("data", [])
                        true_coords = data.get("headers", [])
                        tables.append({
                            "file": filename,
                            "table_data": table,
                            "true_coords": [(h['row'], h['col']) for h in true_coords]
                        })
                except Exception as e:
                    logging.error(f"Ошибка чтения {filename}: {e}")
        logging.info(f"Загружено {len(tables)} JSON-таблиц")
        return tables
    
    def table_to_text(self, table_data: List[List[Any]]) -> str:
        """Конвертация таблицы в текст"""
        if not table_data:
            return ""
        header = "| " + " | ".join(map(str, table_data[0])) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in table_data[1:]]
        return "\n".join([header] + rows)
    
    def create_strict_system_prompt(self) -> str:
        """Жесткий системный промпт для контроля формата"""
        return """You are a precise table header detection system.

CRITICAL RULES:
1. You MUST respond ONLY with valid JSON
2. NO explanations, NO reasoning, NO additional text
3. Use EXACTLY this format: {"headers": [{"row": 0, "col": 0}, {"row": 0, "col": 1}]}
4. "row" and "col" must be integers (0-indexed)
5. Include ALL and ONLY cells that are table headers
6. If uncertain, prefer precision over recall

RESPONSE FORMAT (copy this structure):
{"headers": [{"row": <int>, "col": <int>}]}

EXAMPLES OF VALID RESPONSES:
{"headers": [{"row": 0, "col": 0}, {"row": 0, "col": 1}, {"row": 0, "col": 2}]}
{"headers": [{"row": 0, "col": 0}, {"row": 1, "col": 0}]}
{"headers": []}

DO NOT write anything except the JSON object."""

    def create_user_prompt_with_constraints(self, table_text: str, strategy: str) -> str:
        """User prompt с дополнительными ограничениями формата"""
        
        base_strategies = {
            "zero_shot": """Identify all header cells in this table.
Return ONLY valid JSON in the format: {{"headers": [{{"row": 0, "col": 0}}]}}

Table:
{table}

JSON response:""",
            
            "few_shot": """Examples:
Input: | ID | Name | Age |
Output: {{"headers": [{{"row": 0, "col": 0}}, {{"row": 0, "col": 1}}, {{"row": 0, "col": 2}}]}}

Input: | Product | Price |
       | Electronics | Category |
Output: {{"headers": [{{"row": 0, "col": 0}}, {{"row": 0, "col": 1}}, {{"row": 1, "col": 1}}]}}

Now identify headers for:
{table}

JSON response:""",
            
            "cot": """Analyze this table step-by-step:
1. Examine first row data types
2. Compare with subsequent rows
3. Identify header cells

Table:
{table}

After analysis, respond with ONLY JSON: {{"headers": [{{"row": X, "col": Y}}]}}

JSON response:""",
            
            "role": """You are a data engineer expert. Identify table headers with precision.

Table:
{table}

Respond with ONLY JSON: {{"headers": [{{"row": X, "col": Y}}]}}

JSON response:""",
        }
        
        prompt_template = base_strategies.get(strategy, base_strategies["zero_shot"])
        return prompt_template.format(table=table_text)
    
    def make_api_call(self, model: str, messages: List[Dict], max_retries: int = 3) -> Dict:
        """API вызов с retry логикой"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=500,  # Ограничиваем длину ответа
                    response_format={"type": "json_object"}  # Форсируем JSON (если API поддерживает)
                )
                
                duration = time.time() - start_time
                
                return {
                    "success": True,
                    "response": completion.choices[0].message.content,
                    "duration": duration,
                    "tokens_used": {
                        "prompt": completion.usage.prompt_tokens,
                        "completion": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens
                    } if hasattr(completion, 'usage') else None,
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model}: {e}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def collect_responses(self, models: List[str], prompts: List[Dict[str, str]]):
        """Основной цикл сбора ответов"""
        tables = self.load_json_tables()
        total_tasks = len(models) * len(prompts) * len(tables)
        
        telegram_log(f"🚀 Начало сбора ответов. Всего запросов: {total_tasks}")
        logging.info(f"Starting collection: {len(models)} models × {len(prompts)} prompts × {len(tables)} tables")
        
        processed = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_idx, model in enumerate(models):
            for prompt_idx, prompt_config in enumerate(prompts):
                for table_idx, tbl in enumerate(tables):
                    
                    # Формируем сообщения
                    system_prompt = self.create_strict_system_prompt()
                    user_prompt = self.create_user_prompt_with_constraints(
                        self.table_to_text(tbl['table_data']),
                        prompt_config.get('strategy', 'zero_shot')
                    )
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Делаем запрос
                    result = self.make_api_call(model, messages)
                    
                    # Сохраняем полную информацию
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "model_idx": model_idx,
                        "prompt_strategy": prompt_config.get('strategy', 'unknown'),
                        "prompt_idx": prompt_idx,
                        "table_file": tbl['file'],
                        "table_idx": table_idx,
                        "table_size": f"{len(tbl['table_data'])}x{len(tbl['table_data'][0]) if tbl['table_data'] else 0}",
                        "true_headers": tbl['true_coords'],
                        "true_headers_count": len(tbl['true_coords']),
                        "api_success": result["success"],
                        "raw_response": result.get("response", ""),
                        "error_message": result.get("error", ""),
                        "duration_sec": result.get("duration", 0),
                        "tokens": result.get("tokens_used"),
                        "retry_attempts": result.get("attempt", 1),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt
                    }
                    
                    if result["success"]:
                        self.responses.append(response_data)
                    else:
                        self.failed_requests.append(response_data)
                        logging.error(f"Failed: {model} | {prompt_config['strategy']} | {tbl['file']}")
                    
                    processed += 1
                    
                    # Периодическое сохранение и логирование
                    if processed % 10 == 0:
                        self.save_checkpoint(timestamp)
                        success_rate = len(self.responses) / processed * 100
                        telegram_log(
                            f"📊 Прогресс: {processed}/{total_tasks} ({processed/total_tasks*100:.1f}%)\n"
                            f"✅ Успешных: {len(self.responses)}\n"
                            f"❌ Ошибок: {len(self.failed_requests)}\n"
                            f"📈 Success rate: {success_rate:.1f}%"
                        )
                        logging.info(f"Progress: {processed}/{total_tasks} | Success rate: {success_rate:.1f}%")
                    
                    # Небольшая задержка между запросами
                    time.sleep(0.5)
        
        # Финальное сохранение
        self.save_final_results(timestamp)
        telegram_log(
            f"✅ Сбор завершен!\n"
            f"Всего ответов: {len(self.responses)}\n"
            f"Ошибок: {len(self.failed_requests)}\n"
            f"Success rate: {len(self.responses)/total_tasks*100:.1f}%"
        )
    
    def save_checkpoint(self, timestamp: str):
        """Промежуточное сохранение"""
        checkpoint_file = os.path.join(self.output_dir, f"checkpoint_{timestamp}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "responses": self.responses,
                "failed": self.failed_requests,
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "total_collected": len(self.responses),
                    "total_failed": len(self.failed_requests)
                }
            }, f, ensure_ascii=False, indent=2)
    
    def save_final_results(self, timestamp: str):
        """Финальное сохранение в разных форматах"""
        
        # 1. JSON (полные данные)
        json_file = os.path.join(self.output_dir, f"responses_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "responses": self.responses,
                "failed": self.failed_requests,
                "metadata": {
                    "collection_date": timestamp,
                    "total_collected": len(self.responses),
                    "total_failed": len(self.failed_requests),
                    "success_rate": len(self.responses) / (len(self.responses) + len(self.failed_requests)) * 100
                }
            }, f, ensure_ascii=False, indent=2)
        
        # 2. CSV (для быстрого просмотра)
        if self.responses:
            df = pd.DataFrame(self.responses)
            csv_file = os.path.join(self.output_dir, f"responses_{timestamp}.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 3. Отдельный файл с ошибками
        if self.failed_requests:
            failed_file = os.path.join(self.output_dir, f"failed_{timestamp}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_requests, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Results saved to {self.output_dir}")
        print(f"\n✅ Файлы сохранены в: {self.output_dir}")
        print(f"   - {json_file}")
        if self.responses:
            print(f"   - {csv_file}")
        if self.failed_requests:
            print(f"   - {failed_file}")


# Определение моделей и промптов
MODELS = [
    # Small models
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct",
    "ministral-8b",
    "phi-4",
    "qwen3-8b",
    "glm-4.5-air",
    
    # Medium models
    "gemma-2-27b-it",
    "qwen-2.5-72b-instruct",
    "llama-3.3-70b-instruct",
    "deepseek-chat",
    "mistral-small-3.2-24b-instruct",
    "qwen3-32b",
    "ministral-14b-2512",
]

PROMPT_STRATEGIES = [
    {"strategy": "zero_shot", "name": "Zero-shot"},
    {"strategy": "few_shot", "name": "Few-shot"},
    {"strategy": "cot", "name": "Chain-of-Thought"},
    {"strategy": "role", "name": "Role Prompting"},
]


if __name__ == "__main__":
    collector = ResponseCollector(
        json_dir=r"C:\Users\Юзя\Desktop\better_experiment\Jsons_tables",
        output_dir="raw_responses"
    )
    
    collector.collect_responses(MODELS, PROMPT_STRATEGIES)