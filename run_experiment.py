import argparse
import sys
from pathlib import Path

from data_collector import ResponseCollector
from prompts import PROMPTS
from config import JSON_DIR, OUTPUT_DIR, get_models, MODEL_SET


def print_banner():
    """Печать баннера"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         TABLE HEADER DETECTION EXPERIMENT                    ║
║         Benchmarking LLMs on Structured Data                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Table Header Detection Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-set',
        choices=['small', 'medium', 'all'],
        default=MODEL_SET,
        help='Набор моделей для тестирования (по умолчанию из config.py)'
    )
    
    parser.add_argument(
        '--prompts',
        nargs='+',
        type=int,
        metavar='INDEX',
        help='Индексы промптов для тестирования (0-12). Если не указано, используются все'
    )
    
    parser.add_argument(
        '--list-prompts',
        action='store_true',
        help='Показать список доступных промптов и выйти'
    )
    
    parser.add_argument(
        '--output',
        default=OUTPUT_DIR,
        help=f'Директория для сохранения результатов (по умолчанию: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Тестовый запуск с 2 моделями и 3 промптами'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Пропустить подтверждение'
    )
    
    args = parser.parse_args()
    
    # Показать список промптов
    if args.list_prompts:
        print("\n📝 ДОСТУПНЫЕ ПРОМПТЫ:")
        print("="*70)
        for i, p in enumerate(PROMPTS):
            print(f"{i:2d}. {p['name']}")
        print("="*70)
        print(f"\nВсего промптов: {len(PROMPTS)}")
        print("\nИспользование: python run_experiment.py --prompts 0 1 3")
        return
    
    # Получаем модели
    if args.test:
        models = get_models()[:2]  # Только 2 модели
        selected_prompts = PROMPTS[:3]  # Только 3 промпта
        print("\n🧪 ТЕСТОВЫЙ РЕЖИМ")
    else:
        # Устанавливаем MODEL_SET из аргументов
        import config
        config.MODEL_SET = args.model_set
        models = get_models()
        
        # Выбор промптов
        if args.prompts:
            selected_prompts = []
            for idx in args.prompts:
                if 0 <= idx < len(PROMPTS):
                    selected_prompts.append(PROMPTS[idx])
                else:
                    print(f"⚠️  Предупреждение: Индекс {idx} вне диапазона (0-{len(PROMPTS)-1})")
            
            if not selected_prompts:
                print("❌ Ошибка: Не выбрано ни одного валидного промпта")
                return
        else:
            selected_prompts = PROMPTS
    
    # Вывод конфигурации
    print(f"\n{'='*70}")
    print("КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА")
    print(f"{'='*70}")
    print(f"\n📊 Модели ({len(models)}):")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model}")
    
    print(f"\n📝 Промпты ({len(selected_prompts)}):")
    for i, prompt in enumerate(selected_prompts, 1):
        print(f"  {i:2d}. {prompt['name']}")
    
    # Проверка наличия таблиц
    json_path = Path(JSON_DIR)
    if not json_path.exists():
        print(f"\n❌ ОШИБКА: Директория с таблицами не найдена: {JSON_DIR}")
        print(f"💡 Убедитесь, что путь указан правильно в config.py или переменной окружения JSON_DIR")
        return
    
    if not json_path.is_dir():
        print(f"\n❌ ОШИБКА: {JSON_DIR} не является директорией")
        return
    
    table_files = list(json_path.glob("*.json"))
    if not table_files:
        print(f"\n❌ ОШИБКА: В директории {JSON_DIR} не найдено JSON файлов")
        print(f"💡 Убедитесь, что JSON файлы находятся в указанной директории")
        return
    
    print(f"\n📄 Таблиц: {len(table_files)}")
    
    # Подсчет запросов
    total_requests = len(models) * len(selected_prompts) * len(table_files)
    estimated_time_min = total_requests * 2 / 60  # ~2 сек на запрос
    
    print(f"\n{'='*70}")
    print(f"🎯 Будет выполнено запросов: {total_requests}")
    print(f"⏱️  Примерное время: {estimated_time_min:.1f} минут")
    print(f"💾 Результаты будут сохранены в: {args.output}")
    print(f"{'='*70}")
    
    # Подтверждение
    if not args.yes and not args.test:
        response = input("\n❓ Продолжить? (y/n): ")
        if response.lower() != 'y':
            print("❌ Отменено пользователем")
            return
    
    # Запуск сбора
    print(f"\n{'='*70}")
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА")
    print(f"{'='*70}\n")
    
    try:
        collector = ResponseCollector(
            json_dir=JSON_DIR,
            output_dir=args.output
        )
        
        collector.collect_responses(models, selected_prompts)
        
        print(f"\n{'='*70}")
        print("✅ ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
        print(f"{'='*70}")
        print(f"\n📁 Результаты сохранены в: {args.output}")
        print("\n💡 Следующий шаг: запустите analyzer.py для анализа результатов")
        print(f"   python analyzer.py {args.output}/responses_*.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Эксперимент прерван пользователем")
        print("💾 Промежуточные результаты сохранены в checkpoint файлах")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()