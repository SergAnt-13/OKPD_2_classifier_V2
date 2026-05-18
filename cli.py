# cli.py
import argparse
import sys
from pathlib import Path

from config.settings import TRAINING_DATA_DIR
from src.inference.pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="OKPD-2 Classifier V2")
    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # --- predict-text ---
    pred_text = subparsers.add_parser("predict-text", help="Предсказать код для одного текста")
    pred_text.add_argument("text", type=str, help="Текст товара")

    # --- predict ---
    pred_file = subparsers.add_parser("predict", help="Предсказать коды для файла")
    pred_file.add_argument("--input", type=str, required=True, help="Путь к CSV/Excel")
    pred_file.add_argument("--output", type=str, default=None, help="Куда сохранить результат")
    pred_file.add_argument("--column", type=str, default="Номенклатура", help="Колонка с текстом")

    # --- eda ---
    eda_cmd = subparsers.add_parser("eda", help="Запустить EDA")
    eda_cmd.add_argument("--input", type=str, default=None, help="Путь к файлу с данными")

    # --- debug-search ---
    debug_cmd = subparsers.add_parser("debug-search", help="Показать топ-10 кандидатов для текста")
    debug_cmd.add_argument("text", type=str, help="Текст товара")

    args = parser.parse_args()

    if args.command == "predict-text":
        pipeline = InferencePipeline()
        result = pipeline.predict_single(args.text)
        print(f"Запрос: {result['text']}")
        print(f"Нормализованный: {result['normalized_text']}")
        print()
        print("=" * 60)
        print(" 1. RETRIEVAL (bi-encoder + FAISS)")
        print("=" * 60)
        print(f" Топ-5 кандидатов:")
        for i, cand in enumerate(result['top_candidates'][:5], 1):
            print(f"   {i}. {cand['code']} | {cand['score']:.4f} | {cand['title']}")
        print()
        print("=" * 60)
        print(" 2. CLASSIFIER (RuBERT)")
        print("=" * 60)
        print(f" Код:             {result['classifier_code']}")
        print(f" Уверенность:     {result['classifier_confidence']:.4f}")
        print(f" Margin:          {result['margin']:.4f}")
        print(f" Энтропия:        {result['entropy']:.4f}")
        print()
        print("=" * 60)
        print(" 3. DECISION ENGINE")
        print("=" * 60)
        print(f" Финальный код:   {result['final_prediction']}")
        print(f" Итоговая увер.:  {result['confidence']:.4f}")
        print(f" Роутинг:         {result['routing']}")
        print(f" Причины:         {', '.join(result['reasons'])}")
        print(f" Agreement:       {'Да' if result['agreement'] else 'Нет'}")
        print(f" Hierarchy OK:    {'Да' if result['hierarchy_ok'] else 'Нет'}")


    elif args.command == "predict":

        from datetime import datetime

        import json

        pipeline = InferencePipeline()

        input_path = Path(args.input)

        # Папка запуска с датой и временем

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_dir = Path("runs") / timestamp / "predictions"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Определяем выходной файл

        if args.output:

            output_path = Path(args.output)

        else:

            output_path = input_path.with_suffix(".predicted.xlsx")

        # Если путь не абсолютный, кладём в run_dir

        if not output_path.is_absolute():
            output_path = run_dir / output_path.name

        df = pipeline.predict_file(input_path, text_column=args.column)

        df.to_excel(output_path, index=False)

        # Конфиг запуска (опционально)

        config = {

            "input": str(input_path.absolute()),

            "output": str(output_path.absolute()),

            "text_column": args.column,

            "timestamp": timestamp,

        }

        (run_dir.parent / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

        print(f"Результаты сохранены в {output_path}")

        print(f"Распределение роутинга:\n{df['routing'].value_counts().to_string()}")

    elif args.command == "eda":
        from src.evaluation.eda_report import OKPDEDA
        import pandas as pd

        input_path = Path(args.input) if args.input else TRAINING_DATA_DIR / "train.xlsx"
        output_dir = TRAINING_DATA_DIR / "eda_report"

        df = pd.read_excel(input_path)
        eda = OKPDEDA(df, output_dir=output_dir)
        eda.run_all()

    elif args.command == "debug-search":
        from src.retrieval.retriever import Retriever

        retriever = Retriever()
        # Используем нормализацию из того же cleaner, что и у retriever (с сокращениями)
        query = retriever.cleaner.retrieval_view_normalized(args.text)
        result = retriever.search_normalized(query, top_k=10)

        print(f"\nЗапрос: {args.text}")
        print(f"Нормализованный: {query}\n")
        print(f"{'Ранг':<5} {'Код':<15} {'Родитель':<15} {'Score':<8} Название")
        print("-" * 90)
        for i, cand in enumerate(result["candidates"], 1):
            print(f"{i:<5} {cand['code']:<15} {cand['parent_code']:<15} {cand['score']:.4f}   {cand['title']}")

    else:
        parser.print_help()




if __name__ == "__main__":
    main()