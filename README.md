# Better Experiment

Фреймворк для оценки LLM на задаче распознавания заголовков таблиц. Проект состоит из двух этапов: сбор ответов моделей через Bothub API и последующий анализ с расчётом метрик.

## Возможности

- **data_collector.py** — сбор ответов LLM без немедленного анализа
- **analyzer.py** — парсинг ответов и расчёт Precision, Recall, F1-score
- Многомодельное тестирование (Gemma, Qwen, Llama, Mistral, DeepSeek и др.)
- Четыре стратегии промптов: Zero-shot, Few-shot, Chain-of-Thought, Role Prompting
- Инкрементальное сохранение и Telegram-уведомления о прогрессе
- Визуализация результатов (heatmap, bar charts)

## Установка

```bash
git clone git@github.com:YRL-AIDA/Better_Experiment.git
cd Better_Experiment
pip install -r requirements.txt
```

## Настройка окружения

Все ключи и секреты хранятся только в файле `.env` (он в `.gitignore` и не попадает в репозиторий).

1. Скопируйте пример: `cp .env.example .env` (или создайте `.env` вручную).
2. Заполните переменные в `.env`:

```
BOTHUB_API_KEY=your_bothub_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

`BOTHUB_API_KEY` обязателен для сбора ответов. Telegram — опционально (уведомления о прогрессе).

## Структура проекта

```
Better_Experiment/
├── data_collector.py    # Сбор ответов моделей через Bothub API
├── analyzer.py          # Анализ ответов и расчёт метрик
├── .env.example         # Пример переменных окружения (скопировать в .env)
├── Jsons_tables/        # JSON-таблицы для тестирования
│   ├── PMC493266_tables_cleaned.json
│   └── PMC493271_tables_cleaned.json
├── requirements.txt
└── README.md
```

## Использование

### 1. Сбор ответов

JSON-таблицы должны содержать поля `data` (2D-массив ячеек) и `headers` (координаты заголовков `[{row, col}, ...]`).

```python
from data_collector import ResponseCollector, MODELS, PROMPT_STRATEGIES

collector = ResponseCollector(
    json_dir="Jsons_tables",
    output_dir="raw_responses"
)
collector.collect_responses(MODELS, PROMPT_STRATEGIES)
```

Результаты сохраняются в `raw_responses/`:
- `responses_YYYYMMDD_HHMMSS.json` — полные данные
- `responses_YYYYMMDD_HHMMSS.csv` — табличный вид
- `failed_YYYYMMDD_HHMMSS.json` — неудачные запросы

### 2. Анализ результатов

```python
from analyzer import ResponseAnalyzer

analyzer = ResponseAnalyzer("raw_responses/responses_YYYYMMDD_HHMMSS.json")
analyzer.load_data()
analyzer.process_all_responses()
analyzer.save_analysis_results()
analyzer.create_visualizations()
```

Результаты анализа сохраняются в `analysis_results/`:
- `detailed_metrics_*.xlsx` / `*.csv` — метрики по каждому ответу
- `summary_statistics_*.xlsx` — агрегация по моделям и стратегиям
- `f1_by_model.png`, `heatmap_model_strategy.png` — визуализации

## Поддерживаемые модели

| Размер | Модели |
|--------|--------|
| Small | gemma-2-9b-it, qwen-2.5-7b-instruct, llama-3.1-8b-instruct, mistral-7b-instruct, ministral-8b, phi-4, qwen3-8b, glm-4.5-air |
| Medium | gemma-2-27b-it, qwen-2.5-72b-instruct, llama-3.3-70b-instruct, deepseek-chat, mistral-small-3.2-24b-instruct, qwen3-32b, ministral-14b-2512 |

## Формат JSON-таблиц

Для `data_collector` ожидается формат:

```json
{
  "data": [["Header1", "Header2"], ["val1", "val2"]],
  "headers": [{"row": 0, "col": 0}, {"row": 0, "col": 1}]
}
```

Папка `Jsons_tables/` содержит таблицы в формате PMC (cells с row_nums, column_nums). При необходимости используйте скрипт конвертации в ожидаемый формат.

## Метрики

- **Precision** — доля правильно найденных заголовков среди предсказанных
- **Recall** — доля найденных из всех истинных заголовков
- **F1-score** — гармоническое среднее Precision и Recall

## Лицензия

Проект распространяется под лицензией MIT.
