import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re


class OKPDEDA:

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. DISTRIBUTION OF CLASSES
    # -----------------------------
    def plot_class_distribution(self, top_n=30):

        counts = self.df["Код ОКПД2"].value_counts().head(top_n)

        plt.figure()
        counts.plot(kind="bar")
        plt.title("Top OKPD classes distribution")
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png")
        plt.close()

    # -----------------------------
    # 2. TEXT LENGTH ANALYSIS
    # -----------------------------
    def plot_text_length(self):

        self.df["text_len"] = self.df["Номенклатура"].astype(str).apply(len)

        plt.figure()
        self.df["text_len"].hist(bins=50)
        plt.title("Text length distribution")

        plt.tight_layout()
        plt.savefig(self.output_dir / "text_length.png")
        plt.close()

    # -----------------------------
    # 3. DIRTY PATTERN ANALYSIS
    # -----------------------------
    def dirty_pattern_stats(self):

        patterns = {
            "has_digits": r"\d+",
            "has_caps": r"[A-ZА-Я]{3,}",
            "has_gost": r"ГОСТ|TU|ТУ",
        }

        stats = {}

        for name, pattern in patterns.items():
            stats[name] = self.df["Номенклатура"].astype(str).str.contains(pattern).mean()

        return stats

    # -----------------------------
    # 4. TOP TOKENS
    # -----------------------------
    def top_tokens(self, top_n=30):

        text = " ".join(self.df["Номенклатура"].astype(str).values)
        tokens = re.findall(r"\w+", text.lower())

        counter = Counter(tokens)

        return counter.most_common(top_n)

    # -----------------------------
    # RUN ALL
    # -----------------------------
    def run_all(self):

        print("Running EDA...")

        self.plot_class_distribution()
        self.plot_text_length()

        dirty_stats = self.dirty_pattern_stats()
        top_tokens = self.top_tokens()

        # save summary
        summary_path = self.output_dir / "summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("DIRTY PATTERN STATS\n")
            f.write(str(dirty_stats))
            f.write("\n\nTOP TOKENS\n")
            f.write(str(top_tokens))

        print("EDA completed ->", self.output_dir)