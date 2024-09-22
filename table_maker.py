import pandas as pd
import pprint

result_table = "results/default-model-configuration/regression/regression_table_rmse.csv"


def make_bold(input):
    print("input", input)
    return '("textbf", "--rwrap")'


def make_default_performance_table(df):
    # Set our multi-index rows
    df.index = pd.MultiIndex.from_tuples(
        [
            ("REG", "car"),
            ("REG", "student"),
            ("REG", "productivity"),
            ("REG", "medical"),
            ("REG", "crimes"),
            ("REG", "crab"),
            ("REG", "wine"),
            ("REG", "bike"),
            ("REG", "housing"),
            ("REG", "diamond"),
        ]
    )

    # Convert index columns (task type, dataset name) into columns
    df = df.reset_index()

    # Set our multi-index columns
    df.columns = pd.MultiIndex.from_tuples(
        [
            ("", "", "Task"),
            ("", "", "Dataset"),
            ("Interpretable Models", "GAMs", "GAM-Splines"),
            ("Interpretable Models", "GAMs", "EBM"),
            ("Interpretable Models", "GAMs", "NAM"),
            ("Interpretable Models", "GAMs", "GAMI-Net"),
            ("Interpretable Models", "GAMs", "ExNN"),
            ("Interpretable Models", "GAMs", "IGANN"),
            ("Interpretable Models", "Traditional", "LR"),
            ("Interpretable Models", "Traditional", "DT"),
            ("Black-box Models", "", "RF"),
            ("Black-box Models", "", "XGB"),
            ("Black-box Models", "", "MLP"),
        ]
    )

    # set_table_styles([])
    styler = df.style

    # bold column names
    df.style.applymap_index(make_bold, axis="columns")

    latex_table = styler.to_latex(
        position="htp",
        caption="Predictive performance using default hyperparameters. Classification tasks are assessed using the AUROC, whereas regression tasks are measured using RMSE",
        label="tab:pred_perf_default_hp",
        siunitx=True,
        multicol_align="c",
        multirow_align="c",
        hrules=True,
    )

    return latex_table


if __name__ == "__main__":
    df = pd.read_csv(result_table, index_col=0)
    pprint.pp(df)
    latex_table = make_default_performance_table(df)
    pprint.pp(latex_table.splitlines())
    print(latex_table)
