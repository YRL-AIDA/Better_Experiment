import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ResponseAnalyzer:
    """Анализатор собранных ответов моделей"""
    
    def __init__(self, responses_file: str):
        self.responses_file = responses_file
        self.data = None
        self.metrics = []
        self.parse_failures = []
        
    def load_data(self):
        """Загрузка файла с ответами"""
        try:
            with open(self.responses_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Файл не найден: {self.responses_file}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка парсинга JSON файла {self.responses_file}: {e}")
            raise
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла {self.responses_file}: {e}")
            raise
        
        if not isinstance(self.data, dict):
            raise ValueError(f"Неверный формат данных в файле {self.responses_file}: ожидается словарь")
        
        total_responses = len(self.data.get('responses', []))
        total_failed = len(self.data.get('failed', []))
        
        if total_responses == 0 and total_failed == 0:
            logging.warning(f"Файл {self.responses_file} не содержит данных")
        
        logging.info(f"Loaded {total_responses} successful responses and {total_failed} failed requests")
        print(f"📊 Загружено успешных ответов: {total_responses}")
        print(f"❌ Неудачных запросов: {total_failed}")
        
        if 'metadata' in self.data:
            print(f"\n📋 Метаданные:")
            for key, value in self.data['metadata'].items():
                print(f"   {key}: {value}")
    
    def parse_response(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг ответа модели с множественными стратегиями"""
        if not text or not isinstance(text, str):
            return None
        
        # Стратегии парсинга в порядке приоритета
        strategies = [
            self._parse_clean_json,
            self._parse_json_with_markdown,
            self._parse_json_with_backticks,
            self._parse_coordinate_pairs,
            self._parse_array_format,
            self._parse_text_pattern,
        ]
        
        for strategy in strategies:
            try:
                result = strategy(text)
                if result is not None:
                    return result
            except Exception as e:
                continue
        
        return None
    
    def _parse_clean_json(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг чистого JSON"""
        text = text.strip()
        try:
            data = json.loads(text)
            if "headers" in data and isinstance(data["headers"], list):
                return [
                    (h['row'], h['col']) 
                    for h in data['headers'] 
                    if isinstance(h, dict) and 'row' in h and 'col' in h
                    and isinstance(h['row'], int) and isinstance(h['col'], int)
                ]
        except json.JSONDecodeError:
            return None
    
    def _parse_json_with_markdown(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг JSON в markdown блоках"""
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "headers" in data:
                    return [
                        (h['row'], h['col']) 
                        for h in data['headers'] 
                        if isinstance(h, dict) and 'row' in h and 'col' in h
                    ]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None
        return None
    
    def _parse_json_with_backticks(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг JSON с одинарными backticks"""
        pattern = r'`(\{.*?\})`'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "headers" in data:
                    return [(h['row'], h['col']) for h in data['headers'] 
                            if isinstance(h, dict) and 'row' in h and 'col' in h]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None
        return None
    
    def _parse_coordinate_pairs(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг паттернов row: X, col: Y"""
        pattern = r'["\']?row["\']?\s*:\s*(\d+)[,\s]+["\']?col["\']?\s*:\s*(\d+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return [(int(r), int(c)) for r, c in matches]
        return None
    
    def _parse_array_format(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг формата [[0,1], [0,2]]"""
        pattern = r'\[(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, text)
        if matches:
            return [(int(r), int(c)) for r, c in matches]
        return None
    
    def _parse_text_pattern(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Парсинг текстовых описаний (0,1)"""
        pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(pattern, text)
        if matches:
            return [(int(r), int(c)) for r, c in matches]
        return None
    
    def calculate_metrics(self, pred: List[Tuple], true: List[Tuple]) -> Dict:
        """Расчет метрик качества"""
        set_p = set(pred) if pred else set()
        set_t = set(true) if true else set()
        
        tp = len(set_p & set_t)  # True Positives
        fp = len(set_p - set_t)  # False Positives
        fn = len(set_t - set_p)  # False Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pred_count": len(pred),
            "true_count": len(true),
            "exact_match": 1 if set_p == set_t else 0
        }
    
    def process_all_responses(self):
        """Обработка всех ответов и расчет метрик"""
        if self.data is None:
            raise ValueError("Данные не загружены. Вызовите load_data() сначала.")
        
        responses = self.data.get('responses', [])
        total = len(responses)
        
        if total == 0:
            logging.warning("Нет ответов для обработки")
            return
        
        logging.info(f"Processing {total} responses...")
        
        for idx, resp in enumerate(responses):
            if not isinstance(resp, dict):
                logging.warning(f"Пропуск некорректного ответа #{idx}: не является словарем")
                continue
            
            raw = resp.get('raw_response', '')
            parsed = self.parse_response(raw)
            true_headers = resp.get('true_headers', [])
            
            # Валидация и нормализация true_headers
            true = []
            for h in true_headers:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    try:
                        true.append((int(h[0]), int(h[1])))
                    except (ValueError, TypeError):
                        logging.warning(f"Некорректные координаты заголовка в ответе #{idx}: {h}")
                elif isinstance(h, dict) and 'row' in h and 'col' in h:
                    try:
                        true.append((int(h['row']), int(h['col'])))
                    except (ValueError, TypeError):
                        logging.warning(f"Некорректные координаты заголовка в ответе #{idx}: {h}")
            
            # Определяем успешность парсинга
            parse_success = parsed is not None
            
            if not parse_success:
                self.parse_failures.append({
                    "model": resp.get('model'),
                    "prompt_name": resp.get('prompt_name'),
                    "table_file": resp.get('table_file'),
                    "raw_response": raw[:500]  # Первые 500 символов
                })
            
            # Расчет метрик
            if parsed is not None:
                metrics = self.calculate_metrics(parsed, true)
            else:
                # Если не смогли распарсить, все метрики = 0
                metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(true),
                    "pred_count": 0,
                    "true_count": len(true),
                    "exact_match": 0
                }
            
            self.metrics.append({
                **resp,
                "parsed_headers": parsed if parsed else [],
                "parse_success": parse_success,
                **metrics
            })
            
            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1}/{total} responses")
        
        logging.info(f"✅ Processing complete. {len(self.metrics)} responses processed")
        
        # Статистика парсинга
        parse_success_count = sum(1 for m in self.metrics if m['parse_success'])
        print(f"\n📊 Статистика парсинга:")
        if total > 0:
            print(f"   Успешно: {parse_success_count}/{total} ({parse_success_count/total*100:.1f}%)")
            print(f"   Неудачно: {len(self.parse_failures)} ({len(self.parse_failures)/total*100:.1f}%)")
        else:
            print(f"   Успешно: {parse_success_count}/{total} (0.0%)")
            print(f"   Неудачно: {len(self.parse_failures)} (0.0%)")
    
    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """Генерация сводной статистики"""
        if not self.metrics:
            logging.warning("Нет метрик для генерации статистики")
            return {
                'by_model': pd.DataFrame(),
                'by_prompt': pd.DataFrame(),
                'by_combination': pd.DataFrame(),
                'by_table': pd.DataFrame()
            }
        
        df = pd.DataFrame(self.metrics)
        
        # 1. Статистика по моделям
        model_stats = df.groupby('model').agg({
            'f1': ['mean', 'std', 'min', 'max', 'median'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'accuracy': 'mean',
            'exact_match': 'mean',
            'parse_success': 'mean',
            'duration_sec': ['mean', 'sum'],
        }).round(4)
        
        # 2. Статистика по промптам
        prompt_stats = df.groupby('prompt_name').agg({
            'f1': ['mean', 'std', 'median'],
            'precision': 'mean',
            'recall': 'mean',
            'accuracy': 'mean',
            'exact_match': 'mean',
            'parse_success': 'mean'
        }).round(4)
        
        # 3. Комбинированная статистика (модель + промпт)
        combo_stats = df.groupby(['model', 'prompt_name']).agg({
            'f1': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'exact_match': 'mean',
            'parse_success': 'mean'
        }).round(4).sort_values('f1', ascending=False)
        
        # 4. Статистика по таблицам (сложность)
        table_stats = df.groupby('table_file').agg({
            'f1': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'true_count': 'first'  # Количество истинных заголовков
        }).round(4).sort_values('f1')
        
        return {
            'by_model': model_stats,
            'by_prompt': prompt_stats,
            'by_combination': combo_stats.head(30),
            'by_table': table_stats
        }
    
    def create_visualizations(self, output_dir: str = "analysis_results"):
        """Создание визуализаций"""
        if not self.metrics:
            logging.warning("Нет метрик для создания визуализаций")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        
        if df.empty:
            logging.warning("DataFrame пуст, визуализации не созданы")
            return
        
        # Настройка стиля
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        # 1. F1 Score по моделям
        fig, ax = plt.subplots(figsize=(12, 6))
        model_f1 = df.groupby('model')['f1'].mean().sort_values(ascending=True)
        model_f1.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Average F1 Score', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title('Model Performance Comparison (F1 Score)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_f1_by_model.png", bbox_inches='tight')
        plt.close()
        
        # 2. Parse success rate
        fig, ax = plt.subplots(figsize=(12, 6))
        parse_success = df.groupby('model')['parse_success'].mean().sort_values(ascending=True)
        parse_success.plot(kind='barh', ax=ax, color='forestgreen')
        ax.set_xlabel('Parse Success Rate', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title('Response Parsing Success Rate', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_parse_success_by_model.png", bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap: Model × Prompt Strategy
        pivot = df.pivot_table(values='f1', index='model', columns='prompt_name', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, 
                    cbar_kws={'label': 'F1 Score'})
        ax.set_title('F1 Score Heatmap: Model × Prompt Strategy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prompt Strategy', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_heatmap_model_prompt.png", bbox_inches='tight')
        plt.close()
        
        # 4. Precision vs Recall scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax.scatter(model_data['recall'], model_data['precision'], 
                      label=model, alpha=0.6, s=50)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision vs Recall by Model', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_precision_recall_scatter.png", bbox_inches='tight')
        plt.close()
        
        # 5. Prompt strategy comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        prompt_f1 = df.groupby('prompt_name')['f1'].mean().sort_values(ascending=True)
        prompt_f1.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Average F1 Score', fontsize=12)
        ax.set_ylabel('Prompt Strategy', fontsize=12)
        ax.set_title('Prompt Strategy Performance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_f1_by_prompt.png", bbox_inches='tight')
        plt.close()
        
        # 6. Distribution of F1 scores
        fig, ax = plt.subplots(figsize=(10, 6))
        df['f1'].hist(bins=30, ax=ax, edgecolor='black', color='skyblue')
        ax.axvline(df['f1'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["f1"].mean():.3f}')
        ax.axvline(df['f1'].median(), color='green', linestyle='--', 
                   label=f'Median: {df["f1"].median():.3f}')
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of F1 Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_f1_distribution.png", bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualizations saved to {output_dir}")
        print(f"📊 Визуализации сохранены в: {output_dir}")
    
    def save_analysis_results(self, output_dir: str = "analysis_results"):
        """Сохранение результатов анализа"""
        if not self.metrics:
            logging.warning("Нет метрик для сохранения")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Детальные метрики
        df = pd.DataFrame(self.metrics)
        
        if df.empty:
            logging.warning("DataFrame пуст, результаты не сохранены")
            return
        df.to_excel(f"{output_dir}/detailed_metrics_{timestamp}.xlsx", index=False, engine='openpyxl')
        df.to_csv(f"{output_dir}/detailed_metrics_{timestamp}.csv", index=False, encoding='utf-8-sig')
        
        # 2. Сводная статистика
        summary = self.generate_summary_statistics()
        with pd.ExcelWriter(f"{output_dir}/summary_statistics_{timestamp}.xlsx", engine='openpyxl') as writer:
            summary['by_model'].to_excel(writer, sheet_name='By Model')
            summary['by_prompt'].to_excel(writer, sheet_name='By Prompt')
            summary['by_combination'].to_excel(writer, sheet_name='Top Combinations')
            summary['by_table'].to_excel(writer, sheet_name='By Table')
        
        # 3. Ошибки парсинга
        if self.parse_failures:
            parse_failures_file = f"{output_dir}/parse_failures_{timestamp}.json"
            with open(parse_failures_file, 'w', encoding='utf-8') as f:
                json.dump(self.parse_failures, f, indent=2, ensure_ascii=False)
            logging.info(f"Parse failures saved: {parse_failures_file}")
        
        # 4. Текстовая сводка
        summary_text = f"{output_dir}/summary_{timestamp}.txt"
        with open(summary_text, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Responses Analyzed: {len(self.metrics)}\n")
            metrics_count = len(self.metrics)
            if metrics_count > 0:
                parse_rate = sum(m['parse_success'] for m in self.metrics) / metrics_count * 100
                f.write(f"Parse Success Rate: {parse_rate:.2f}%\n\n")
            else:
                f.write(f"Parse Success Rate: 0.00%\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  Mean F1 Score: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}\n")
            f.write(f"  Median F1 Score: {df['f1'].median():.4f}\n")
            f.write(f"  Mean Precision: {df['precision'].mean():.4f}\n")
            f.write(f"  Mean Recall: {df['recall'].mean():.4f}\n")
            f.write(f"  Mean Accuracy: {df['accuracy'].mean():.4f}\n")
            f.write(f"  Exact Match Rate: {df['exact_match'].mean()*100:.2f}%\n\n")
            
            f.write("TOP 5 MODELS (by F1):\n")
            top_models = summary['by_model']['f1']['mean'].sort_values(ascending=False).head()
            for i, (model, score) in enumerate(top_models.items(), 1):
                f.write(f"  {i}. {model}: {score:.4f}\n")
            
            f.write("\nTOP 5 PROMPTS (by F1):\n")
            top_prompts = summary['by_prompt']['f1']['mean'].sort_values(ascending=False).head()
            for i, (prompt, score) in enumerate(top_prompts.items(), 1):
                f.write(f"  {i}. {prompt}: {score:.4f}\n")
            
            f.write("\nTOP 10 COMBINATIONS (Model + Prompt):\n")
            for i, ((model, prompt), row) in enumerate(summary['by_combination'].head(10).iterrows(), 1):
                f.write(f"  {i}. {model} + {prompt}: F1={row['f1']:.4f}\n")
        
        logging.info(f"Analysis results saved to {output_dir}")
        print(f"\n✅ РЕЗУЛЬТАТЫ АНАЛИЗА СОХРАНЕНЫ:")
        print(f"   📊 {output_dir}/detailed_metrics_{timestamp}.xlsx")
        print(f"   📈 {output_dir}/summary_statistics_{timestamp}.xlsx")
        print(f"   📝 {output_dir}/summary_{timestamp}.txt")
        
        # Вывод краткой статистики
        print(f"\n{'='*80}")
        print("КРАТКАЯ СТАТИСТИКА")
        print(f"{'='*80}")
        print(f"📊 Всего проанализировано: {len(self.metrics)} ответов")
        metrics_count = len(self.metrics)
        if metrics_count > 0:
            parse_success = sum(m['parse_success'] for m in self.metrics)
            parse_rate = parse_success / metrics_count * 100
            print(f"✅ Успешно распарсено: {parse_success} ({parse_rate:.1f}%)")
        else:
            print(f"✅ Успешно распарсено: 0 (0.0%)")
        print(f"\n📈 Средние метрики:")
        print(f"   F1 Score:  {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
        print(f"   Precision: {df['precision'].mean():.4f}")
        print(f"   Recall:    {df['recall'].mean():.4f}")
        print(f"   Accuracy:  {df['accuracy'].mean():.4f}")
        print(f"\n🏆 Лучшая модель: {summary['by_model']['f1']['mean'].idxmax()} (F1: {summary['by_model']['f1']['mean'].max():.4f})")
        print(f"🏆 Лучшая стратегия: {summary['by_prompt']['f1']['mean'].idxmax()} (F1: {summary['by_prompt']['f1']['mean'].max():.4f})")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <responses_file.json>")
        print("\nИли укажите путь к файлу напрямую:")
        
        # Автопоиск последнего файла
        responses_dir = Path("raw_responses")
        if responses_dir.exists():
            json_files = sorted(responses_dir.glob("responses_*.json"), 
                              key=lambda x: x.stat().st_mtime, reverse=True)
            if json_files:
                latest_file = json_files[0]
                print(f"\nНайден последний файл: {latest_file}")
                response = input("Использовать его? (y/n): ")
                if response.lower() == 'y':
                    responses_file = str(latest_file)
                else:
                    sys.exit(1)
            else:
                print("Файлы с ответами не найдены")
                sys.exit(1)
        else:
            print("Директория raw_responses не найдена")
            sys.exit(1)
    else:
        responses_file = sys.argv[1]
    
    analyzer = ResponseAnalyzer(responses_file)
    analyzer.load_data()
    analyzer.process_all_responses()
    analyzer.save_analysis_results()
    analyzer.create_visualizations()
