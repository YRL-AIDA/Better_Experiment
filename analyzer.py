import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ResponseAnalyzer:
    """Анализ собранных ответов"""
    
    def __init__(self, responses_file: str):
        self.responses_file = responses_file
        self.data = None
        self.parsed_responses = []
        self.metrics = []
        
    def load_data(self):
        """Загрузка собранных ответов"""
        with open(self.responses_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Загружено {len(self.data['responses'])} ответов")
        
    def parse_response(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг ответа модели с множественными стратегиями"""
        if not text:
            return []
        
        strategies = [
            self._parse_clean_json,
            self._parse_json_with_markdown,
            self._parse_coordinate_pairs,
            self._parse_array_format,
            self._parse_text_pattern
        ]
        
        for strategy in strategies:
            result = strategy(text)
            if result is not None:
                return result
        
        # Если ничего не сработало
        return []
    
    def _parse_clean_json(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг чистого JSON"""
        try:
            # Удаляем возможные пробелы и переносы
            text = text.strip()
            data = json.loads(text)
            if "headers" in data:
                return [(h['row'], h['col']) for h in data['headers'] 
                        if isinstance(h, dict) and 'row' in h and 'col' in h]
        except:
            return None
    
    def _parse_json_with_markdown(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг JSON в markdown блоках"""
        try:
            # Ищем ```json ... ``` или ``` ... ```
            pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if "headers" in data:
                    return [(h['row'], h['col']) for h in data['headers'] 
                            if isinstance(h, dict) and 'row' in h and 'col' in h]
        except:
            return None
    
    def _parse_coordinate_pairs(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг паттернов типа row: 0, col: 1"""
        try:
            pattern = r'["\']?row["\']?\s*:\s*(\d+)[,\s]+["\']?col["\']?\s*:\s*(\d+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return [(int(r), int(c)) for r, c in matches]
        except:
            return None
    
    def _parse_array_format(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг формата [[0,1], [0,2]]"""
        try:
            pattern = r'\[(\d+),\s*(\d+)\]'
            matches = re.findall(pattern, text)
            if matches:
                return [(int(r), int(c)) for r, c in matches]
        except:
            return None
    
    def _parse_text_pattern(self, text: str) -> List[Tuple[int, int]]:
        """Парсинг текстовых описаний типа (0,1)"""
        try:
            pattern = r'\((\d+),\s*(\d+)\)'
            matches = re.findall(pattern, text)
            if matches:
                return [(int(r), int(c)) for r, c in matches]
        except:
            return None
    
    def calculate_metrics(self, pred: List[Tuple], true: List[Tuple]) -> Dict:
        """Расчет метрик"""
        set_p, set_t = set(pred), set(true)
        
        tp = len(set_p & set_t)
        fp = len(set_p - set_t)
        fn = len(set_t - set_p)
        tn = 0  # Для таблиц сложно определить истинно негативные
        
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
            "true_count": len(true)
        }
    
    def analyze_parse_success(self) -> Dict:
        """Анализ успешности парсинга ответов"""
        parse_stats = {
            "total": len(self.data['responses']),
            "successfully_parsed": 0,
            "empty_responses": 0,
            "parse_failures": 0,
            "by_model": {},
            "by_strategy": {}
        }
        
        for resp in self.data['responses']:
            raw = resp['raw_response']
            parsed = self.parse_response(raw)
            
            model = resp['model']
            strategy = resp['prompt_strategy']
            
            # Инициализация счетчиков
            if model not in parse_stats["by_model"]:
                parse_stats["by_model"][model] = {"success": 0, "fail": 0, "empty": 0}
            if strategy not in parse_stats["by_strategy"]:
                parse_stats["by_strategy"][strategy] = {"success": 0, "fail": 0, "empty": 0}
            
            if not raw or raw.strip() == "":
                parse_stats["empty_responses"] += 1
                parse_stats["by_model"][model]["empty"] += 1
                parse_stats["by_strategy"][strategy]["empty"] += 1
            elif len(parsed) > 0:
                parse_stats["successfully_parsed"] += 1
                parse_stats["by_model"][model]["success"] += 1
                parse_stats["by_strategy"][strategy]["success"] += 1
            else:
                parse_stats["parse_failures"] += 1
                parse_stats["by_model"][model]["fail"] += 1
                parse_stats["by_strategy"][strategy]["fail"] += 1
        
        return parse_stats
    
    def process_all_responses(self):
        """Обработка всех ответов и расчет метрик"""
        for resp in self.data['responses']:
            parsed = self.parse_response(resp['raw_response'])
            true = [tuple(h) for h in resp['true_headers']]
            
            metrics = self.calculate_metrics(parsed, true)
            
            self.metrics.append({
                **resp,
                "parsed_headers": parsed,
                "parsed_count": len(parsed),
                "parse_success": len(parsed) > 0,
                **metrics
            })
        
        print(f"Обработано {len(self.metrics)} ответов")
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Генерация сводной статистики"""
        df = pd.DataFrame(self.metrics)
        
        # Группировка по моделям
        model_stats = df.groupby('model').agg({
            'f1': ['mean', 'std', 'min', 'max', 'median'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'accuracy': 'mean',
            'parse_success': 'mean',
            'duration_sec': ['mean', 'sum'],
        }).round(4)
        
        # Группировка по стратегиям промптов
        strategy_stats = df.groupby('prompt_strategy').agg({
            'f1': ['mean', 'std', 'median'],
            'precision': 'mean',
            'recall': 'mean',
            'parse_success': 'mean'
        }).round(4)
        
        # Топ комбинации модель+стратегия
        combo_stats = df.groupby(['model', 'prompt_strategy']).agg({
            'f1': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'parse_success': 'mean'
        }).round(4).sort_values('f1', ascending=False)
        
        return {
            'by_model': model_stats,
            'by_strategy': strategy_stats,
            'by_combination': combo_stats.head(20)
        }
    
    def create_visualizations(self, output_dir: str = "analysis_results"):
        """Создание визуализаций"""
        Path(output_dir).mkdir(exist_ok=True)
        df = pd.DataFrame(self.metrics)
        
        # 1. F1 Score по моделям
        plt.figure(figsize=(14, 6))
        model_f1 = df.groupby('model')['f1'].mean().sort_values(ascending=False)
        plt.barh(range(len(model_f1)), model_f1.values)
        plt.yticks(range(len(model_f1)), model_f1.index)
        plt.xlabel('Average F1 Score')
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_by_model.png", dpi=300)
        plt.close()
        
        # 2. Success rate парсинга
        plt.figure(figsize=(10, 6))
        parse_success = df.groupby('model')['parse_success'].mean().sort_values(ascending=False)
        plt.bar(range(len(parse_success)), parse_success.values)
        plt.xticks(range(len(parse_success)), parse_success.index, rotation=45, ha='right')
        plt.ylabel('Parse Success Rate')
        plt.title('Response Parsing Success Rate by Model')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parse_success_by_model.png", dpi=300)
        plt.close()
        
        # 3. Heatmap стратегий и моделей
        pivot = df.pivot_table(values='f1', index='model', columns='prompt_strategy', aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')
        plt.title('F1 Score: Model × Prompt Strategy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_model_strategy.png", dpi=300)
        plt.close()
        
        print(f"Визуализации сохранены в {output_dir}")
    
    def save_analysis_results(self, output_dir: str = "analysis_results"):
        """Сохранение результатов анализа"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Детальные метрики
        df = pd.DataFrame(self.metrics)
        df.to_excel(f"{output_dir}/detailed_metrics_{timestamp}.xlsx", index=False)
        df.to_csv(f"{output_dir}/detailed_metrics_{timestamp}.csv", index=False)
        
        # 2. Сводная статистика
        summary = self.generate_summary_statistics()
        with pd.ExcelWriter(f"{output_dir}/summary_statistics_{timestamp}.xlsx") as writer:
            summary['by_model'].to_excel(writer, sheet_name='By Model')
            summary['by_strategy'].to_excel(writer, sheet_name='By Strategy')
            summary['by_combination'].to_excel(writer, sheet_name='Top Combinations')
        
        # 3. Статистика парсинга
        parse_stats = self.analyze_parse_success()
        with open(f"{output_dir}/parse_statistics_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(parse_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты анализа сохранены в {output_dir}")
        
        # Вывод краткой статистики
        print("\n=== КРАТКАЯ СТАТИСТИКА ===")
        print(f"Всего проанализировано: {len(self.metrics)} ответов")
        print(f"Успешно распарсено: {sum(1 for m in self.metrics if m['parse_success'])} ({sum(1 for m in self.metrics if m['parse_success'])/len(self.metrics)*100:.1f}%)")
        print(f"\nСредние метрики:")
        print(f"  F1 Score: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
        print(f"  Precision: {df['precision'].mean():.4f}")
        print(f"  Recall: {df['recall'].mean():.4f}")
        print(f"\nЛучшая модель по F1: {summary['by_model']['f1']['mean'].idxmax()} ({summary['by_model']['f1']['mean'].max():.4f})")
        print(f"Лучшая стратегия: {summary['by_strategy']['f1']['mean'].idxmax()} ({summary['by_strategy']['f1']['mean'].max():.4f})")


if __name__ == "__main__":
    # Укажите путь к файлу с ответами
    responses_file = "raw_responses/responses_20250201_143022.json"  # Замените на актуальный
    
    analyzer = ResponseAnalyzer(responses_file)
    analyzer.load_data()
    analyzer.process_all_responses()
    analyzer.save_analysis_results()
    analyzer.create_visualizations()
