import argparse

from src.common.logger import get_logger

logger = get_logger()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "predict",
            "train-bi_encoder",
            "train-classifier",
            "build-faiss",
            "evaluate",
            "eda"
        ]
    )

    args = parser.parse_args()

    logger.info(f"Running command: {args.command}")

    if args.command == "predict":
        print("Prediction pipeline")

    elif args.command == "train-bi_encoder":
        print("Training bi-encoder")

    elif args.command == "train-classifier":
        print("Training classifier")

    elif args.command == "build-faiss":
        print("Building FAISS index")

    elif args.command == "evaluate":
        print("Evaluation pipeline")

    elif args.command == "eda":
        from src.evaluation.eda_report import OKPDEDA
        import pandas as pd
        from config.settings import TRAINING_DATA_DIR

        df = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx")

        eda = OKPDEDA(df, output_dir=TRAINING_DATA_DIR / "eda_report")
        eda.run_all()

if __name__ == "__main__":
    main()