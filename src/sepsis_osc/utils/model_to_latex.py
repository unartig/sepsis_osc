import equinox as eqx
import jax
import pandas as pd

from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel


def clean_name(parts: list[str]) -> str:
    cleaned = []

    for p in parts:
        clean_p = str(p).strip("'").lstrip(".")

        if clean_p == "":
            continue

        # turn [0] into 0
        if clean_p.startswith("[") and clean_p.endswith("]"):
            clean_p = "linear" + clean_p[1:-1]

        cleaned.append(clean_p)

    # keep last two meaningful parts
    if len(cleaned) >= 2:
        cleaned[1] = "GRU " + cleaned[1] if cleaned[1].endswith(("hh", "ih")) else cleaned[1]
        return f"{cleaned[-2]} {cleaned[-1]}"
    if cleaned:
        print("cleand1", cleaned)
        return cleaned[-1]
    return ""


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def generate_param_df(model: LatentDynamicsModel) -> pd.DataFrame:
    rows = []

    for path, value in jax.tree_util.tree_leaves_with_path(model):
        if not eqx.is_array(value):
            continue

        parts = [str(p) for p in path]
        full_path = ".".join(parts)

        if "rollout" in full_path:
            continue
        # Module grouping
        if "latent_pre_encoder" in full_path:
            group = "SOFA Module (Encoder)"
        elif any(k in full_path for k in ["latent_encoder", "latent_rollout", "latent_proj_out"]):
            group = "SOFA Module (RNN)"
        elif any(k in full_path for k in ["inf_encoder", "inf_rollout", "inf_proj_out"]):
            group = "Infection Module"
        elif "decoder" in full_path:
            group = "Decoder Module"
        else:
            group = "General"

        display_name = clean_name(parts)

        rows.append(
            {
                "Module": group,
                "Name": display_name,
                "Shape": str(value.shape),
                "Count": int(value.size),
            }
        )

    return pd.DataFrame(rows)


def df_to_latex_grouped(df: pd.DataFrame, module_order: list[str]) -> str:
    latex = []

    latex.append(r"\begin{tabular}{lcr}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Name} & \textbf{Shape} & \textbf{Count} \\")
    latex.append(r"\hline")

    for module in module_order:
        group = df[df["Module"] == module]
        if group.empty:
            continue

        latex.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{module}}}}} \\")

        total = 0

        for _, row in group.iterrows():
            name = latex_escape(str(row["Name"]))
            shape = row["Shape"]
            count = row["Count"]
            total += count

            latex.append(f"{name} & {shape} & {count:,} \\\\")

        latex.append(rf"\multicolumn{{2}}{{r}}{{\textbf{{Total:}}}} & \textbf{{{total:,}}} \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")

    return "\n".join(latex)


def model_to_latex_figure(model: LatentDynamicsModel) -> str:
    df = generate_param_df(model)

    module_order = [
        "Infection Module",
        "Decoder Module",
        "General",
        "SOFA Module (Encoder)",
        "SOFA Module (RNN)",
    ]

    # split modules into two columns
    mid = len(module_order) // 2
    left_modules = module_order[:mid]
    right_modules = module_order[mid:]

    left_df = df[df["Module"].isin(left_modules)]
    right_df = df[df["Module"].isin(right_modules)]

    left_table = df_to_latex_grouped(left_df, left_modules)
    right_table = df_to_latex_grouped(right_df, right_modules)

    final = f"""
\\begin{{figure*}}[t]
\\centering
\\begin{{minipage}}{{0.48\\textwidth}}
{left_table}
\\end{{minipage}}
\\hfill
\\begin{{minipage}}{{0.48\\textwidth}}
{right_table}
\\end{{minipage}}
\\caption{{Detailed parameter count of the LDM modules.}}
\\end{{figure*}}
            """

    return final
