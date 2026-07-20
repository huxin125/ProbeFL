import os
import argparse
import pandas as pd


# Default prices: USD per 1M tokens.
# Please verify/update these prices according to the official API pricing
# at the time of reporting.
DEFAULT_PRICE_TABLE = {
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-5.4-2026-03-05": {
        "input": 2.50,
        "output": 15.00,
    },
}


def to_numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def filter_valid_rows(df, drop_error_rows=False):
    if not drop_error_rows:
        return df

    if "error_occurred" not in df.columns:
        return df

    s = df["error_occurred"]

    mask_error = (
        (s == True) |
        (s == 1) |
        (s.astype(str).str.lower().isin(["true", "1", "yes"]))
    )

    return df[~mask_error].copy()


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(df.columns)
        )


def build_price_table(args):
    price_table = DEFAULT_PRICE_TABLE.copy()

    # Optional override for the main model.
    if args.main_model is not None:
        if args.main_input_price is None or args.main_output_price is None:
            raise ValueError(
                "If --main-model is provided, both --main-input-price and "
                "--main-output-price must be provided."
            )
        price_table[args.main_model] = {
            "input": args.main_input_price,
            "output": args.main_output_price,
        }

    # Optional override for the verifier model.
    if args.verifier_model is not None:
        if args.verifier_input_price is None or args.verifier_output_price is None:
            raise ValueError(
                "If --verifier-model is provided, both --verifier-input-price "
                "and --verifier-output-price must be provided."
            )
        price_table[args.verifier_model] = {
            "input": args.verifier_input_price,
            "output": args.verifier_output_price,
        }

    return price_table


def get_price(price_table, model_name, token_type):
    if model_name not in price_table:
        raise ValueError(
            f"Unknown model price for model: {model_name}\n"
            f"Known models: {list(price_table.keys())}\n"
            "Please add the model to DEFAULT_PRICE_TABLE or provide prices "
            "through command-line arguments."
        )

    return price_table[model_name][token_type]


def compute_cost(df, price_table):
    required_cols = [
        "bug_id",
        "project",
        "main_model",
        "verifier_model",
        "main_prompt_tokens",
        "main_completion_tokens",
        "main_total_tokens",
        "sub_prompt_tokens",
        "sub_completion_tokens",
        "sub_total_tokens",
    ]
    require_columns(df, required_cols)

    cost_df = df.copy()

    token_cols = [
        "main_prompt_tokens",
        "main_completion_tokens",
        "main_total_tokens",
        "sub_prompt_tokens",
        "sub_completion_tokens",
        "sub_total_tokens",
        "total_tokens",
    ]

    for col in token_cols:
        if col in cost_df.columns:
            cost_df[col] = to_numeric(cost_df, col)

    cost_df["main_input_price_per_1m"] = cost_df["main_model"].apply(
        lambda m: get_price(price_table, m, "input")
    )
    cost_df["main_output_price_per_1m"] = cost_df["main_model"].apply(
        lambda m: get_price(price_table, m, "output")
    )
    cost_df["verifier_input_price_per_1m"] = cost_df["verifier_model"].apply(
        lambda m: get_price(price_table, m, "input")
    )
    cost_df["verifier_output_price_per_1m"] = cost_df["verifier_model"].apply(
        lambda m: get_price(price_table, m, "output")
    )

    cost_df["static_search_cost_usd"] = (
        cost_df["main_prompt_tokens"] / 1_000_000.0
        * cost_df["main_input_price_per_1m"]
        + cost_df["main_completion_tokens"] / 1_000_000.0
        * cost_df["main_output_price_per_1m"]
    )

    cost_df["runtime_probing_cost_usd"] = (
        cost_df["sub_prompt_tokens"] / 1_000_000.0
        * cost_df["verifier_input_price_per_1m"]
        + cost_df["sub_completion_tokens"] / 1_000_000.0
        * cost_df["verifier_output_price_per_1m"]
    )

    cost_df["total_cost_usd"] = (
        cost_df["static_search_cost_usd"]
        + cost_df["runtime_probing_cost_usd"]
    )

    return cost_df


def stage_summary(cost_df):
    rows = []

    rows.append({
        "stage": "Static Search",
        "model": ", ".join(sorted(cost_df["main_model"].unique())),
        "bugs": len(cost_df),
        "prompt_tokens_mean": cost_df["main_prompt_tokens"].mean(),
        "completion_tokens_mean": cost_df["main_completion_tokens"].mean(),
        "total_tokens_mean": cost_df["main_total_tokens"].mean(),
        "cost_mean_usd": cost_df["static_search_cost_usd"].mean(),
        "cost_median_usd": cost_df["static_search_cost_usd"].median(),
        "cost_sum_usd": cost_df["static_search_cost_usd"].sum(),
    })

    rows.append({
        "stage": "Runtime Probing",
        "model": ", ".join(sorted(cost_df["verifier_model"].unique())),
        "bugs": len(cost_df),
        "prompt_tokens_mean": cost_df["sub_prompt_tokens"].mean(),
        "completion_tokens_mean": cost_df["sub_completion_tokens"].mean(),
        "total_tokens_mean": cost_df["sub_total_tokens"].mean(),
        "cost_mean_usd": cost_df["runtime_probing_cost_usd"].mean(),
        "cost_median_usd": cost_df["runtime_probing_cost_usd"].median(),
        "cost_sum_usd": cost_df["runtime_probing_cost_usd"].sum(),
    })

    rows.append({
        "stage": "Total",
        "model": "--",
        "bugs": len(cost_df),
        "prompt_tokens_mean": (
            cost_df["main_prompt_tokens"] + cost_df["sub_prompt_tokens"]
        ).mean(),
        "completion_tokens_mean": (
            cost_df["main_completion_tokens"] + cost_df["sub_completion_tokens"]
        ).mean(),
        "total_tokens_mean": (
            cost_df["main_total_tokens"] + cost_df["sub_total_tokens"]
        ).mean(),
        "cost_mean_usd": cost_df["total_cost_usd"].mean(),
        "cost_median_usd": cost_df["total_cost_usd"].median(),
        "cost_sum_usd": cost_df["total_cost_usd"].sum(),
    })

    return pd.DataFrame(rows)


def project_summary(cost_df):
    grouped = cost_df.groupby("project").agg(
        bugs=("bug_id", "count"),

        main_prompt_tokens_mean=("main_prompt_tokens", "mean"),
        main_completion_tokens_mean=("main_completion_tokens", "mean"),
        main_total_tokens_mean=("main_total_tokens", "mean"),

        sub_prompt_tokens_mean=("sub_prompt_tokens", "mean"),
        sub_completion_tokens_mean=("sub_completion_tokens", "mean"),
        sub_total_tokens_mean=("sub_total_tokens", "mean"),

        static_search_cost_mean_usd=("static_search_cost_usd", "mean"),
        runtime_probing_cost_mean_usd=("runtime_probing_cost_usd", "mean"),
        total_cost_mean_usd=("total_cost_usd", "mean"),

        static_search_cost_sum_usd=("static_search_cost_usd", "sum"),
        runtime_probing_cost_sum_usd=("runtime_probing_cost_usd", "sum"),
        total_cost_sum_usd=("total_cost_usd", "sum"),
    ).reset_index()

    return grouped


def print_stage_summary(summary_df):
    print("\n[Stage-level Cost Summary]")
    for _, row in summary_df.iterrows():
        print(f"\nStage: {row['stage']}")
        print(f"  Model: {row['model']}")
        print(f"  Bugs: {int(row['bugs'])}")
        print(f"  Prompt tokens mean: {row['prompt_tokens_mean'] / 1000:.2f}K")
        print(f"  Completion tokens mean: {row['completion_tokens_mean'] / 1000:.2f}K")
        print(f"  Total tokens mean: {row['total_tokens_mean'] / 1000:.2f}K")
        print(f"  Cost mean: ${row['cost_mean_usd']:.4f} per bug")
        print(f"  Cost median: ${row['cost_median_usd']:.4f} per bug")
        print(f"  Cost sum: ${row['cost_sum_usd']:.2f}")


def print_latex_table_rows(summary_df):
    print("\n[LaTeX Table Rows]")
    for _, row in summary_df.iterrows():
        stage = row["stage"]
        model = row["model"]

        prompt_k = row["prompt_tokens_mean"] / 1000.0
        completion_k = row["completion_tokens_mean"] / 1000.0
        total_k = row["total_tokens_mean"] / 1000.0
        cost = row["cost_mean_usd"]

        print(
            f"{stage} & {model} & "
            f"{prompt_k:.2f}K & {completion_k:.2f}K & "
            f"{total_k:.2f}K & \\${cost:.4f} \\\\"
        )


def save_outputs(cost_df, stage_df, project_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    per_bug_path = os.path.join(out_dir, "rq3_cost_per_bug.csv")
    stage_path = os.path.join(out_dir, "rq3_cost_stage_summary.csv")
    project_path = os.path.join(out_dir, "rq3_cost_project_summary.csv")

    keep_cols = [
        "bug_id",
        "project",
        "main_model",
        "verifier_model",

        "main_prompt_tokens",
        "main_completion_tokens",
        "main_total_tokens",

        "sub_prompt_tokens",
        "sub_completion_tokens",
        "sub_total_tokens",

        "static_search_cost_usd",
        "runtime_probing_cost_usd",
        "total_cost_usd",
    ]

    existing_keep_cols = [c for c in keep_cols if c in cost_df.columns]

    cost_df[existing_keep_cols].to_csv(per_bug_path, index=False)
    stage_df.to_csv(stage_path, index=False)
    project_df.to_csv(project_path, index=False)

    print("\n[Saved]")
    print(per_bug_path)
    print(stage_path)
    print(project_path)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate RQ3 API cost from ProbeFL per-bug token logs."
    )

    parser.add_argument(
        "--per-bug",
        default="./rq3/probefl_rq3_full_per_bug_details.csv",
        help="Input per-bug RQ3 CSV."
    )

    parser.add_argument(
        "--out-dir",
        default="./rq3/cost",
        help="Output directory."
    )

    parser.add_argument(
        "--drop-error-rows",
        action="store_true",
        help="Drop rows where error_occurred is true."
    )

    # Optional custom model prices.
    parser.add_argument(
        "--main-model",
        default=None,
        help="Main model name for overriding price."
    )
    parser.add_argument(
        "--main-input-price",
        type=float,
        default=None,
        help="Main model input price in USD per 1M tokens."
    )
    parser.add_argument(
        "--main-output-price",
        type=float,
        default=None,
        help="Main model output price in USD per 1M tokens."
    )

    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Verifier model name for overriding price."
    )
    parser.add_argument(
        "--verifier-input-price",
        type=float,
        default=None,
        help="Verifier model input price in USD per 1M tokens."
    )
    parser.add_argument(
        "--verifier-output-price",
        type=float,
        default=None,
        help="Verifier model output price in USD per 1M tokens."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.per_bug)
    df = filter_valid_rows(df, drop_error_rows=args.drop_error_rows)

    if df.empty:
        raise ValueError("No rows left after filtering.")

    price_table = build_price_table(args)

    print("[Input]")
    print(f"Rows: {len(df)}")
    print(f"Projects: {sorted(df['project'].unique())}")
    print(f"Main models: {sorted(df['main_model'].unique())}")
    print(f"Verifier models: {sorted(df['verifier_model'].unique())}")

    cost_df = compute_cost(df, price_table)
    stage_df = stage_summary(cost_df)
    project_df = project_summary(cost_df)

    print_stage_summary(stage_df)

    print("\n[Project-level Cost Summary]")
    print(project_df.to_string(index=False))

    print_latex_table_rows(stage_df)

    save_outputs(cost_df, stage_df, project_df, args.out_dir)


if __name__ == "__main__":
    main()
