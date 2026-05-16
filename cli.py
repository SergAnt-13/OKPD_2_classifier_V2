import argparse

from src.common.logger import get_logger

logger = get_logger()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "predict",
            "train-biencoder",
            "train-classifier",
            "build-faiss",
            "evaluate"
        ]
    )

    args = parser.parse_args()

    logger.info(f"Running command: {args.command}")

    if args.command == "predict":
        print("Prediction pipeline")

    elif args.command == "train-biencoder":
        print("Training bi-encoder")

    elif args.command == "train-classifier":
        print("Training classifier")

    elif args.command == "build-faiss":
        print("Building FAISS index")

    elif args.command == "evaluate":
        print("Evaluation pipeline")


if __name__ == "__main__":
    main()