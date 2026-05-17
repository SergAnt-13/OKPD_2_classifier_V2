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

    args = parser.parse_args()

    if args.command == "predict-text":
        pipeline = InferencePipeline()
        result = pipeline.predict_single(args.text)
        print(f"Текст: {result['text']}")
        print(f"Код: {result['final_prediction']}")
        print(f"Уверенность: {result['confidence']:.4f}")
        print(f"Роутинг: {result['routing']}")
        print(f"Причины: {', '.join(result['reasons'])}")

    elif args.command == "predict":
        pipeline = InferencePipeline()
        input_path = Path(args.input)
        df = pipeline.predict_file(input_path, text_column=args.column)

        output_path = args.output or input_path.with_suffix(".predicted.xlsx")
        df.to_excel(output_path, index=False)
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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()