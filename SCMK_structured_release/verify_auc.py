"""Recompute the manuscript AUC table directly from cached sample scores."""
from utils.config import ROOT, load_experiment_configs
from utils.evaluation import summarize_reference_scores


def main():
    configs = load_experiment_configs()
    table = summarize_reference_scores(ROOT, configs)
    table["mean_match"] = (table["mean"].round(3) == table["table_mean"].round(3))
    table["std_match"] = (table["std"].round(3) == table["table_std"].round(3))
    print(table.to_string(index=False))
    print(f"\nMean AUC across datasets: {table['mean'].mean():.12f}")
    ok = bool(table[["mean_match", "std_match"]].to_numpy().all())
    print(f"All manuscript rows match: {ok}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
