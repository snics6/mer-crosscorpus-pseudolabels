"""地図の途中結果をまとめる（`exp/map.py` の結果を読む）。**保存済みの反復の分だけで作る。**

    ./.venv/bin/python exp/summarize_map.py                    # results/map_audio_wavlm/
    ./.venv/bin/python exp/summarize_map.py --dir <フォルダ>

出力（同じフォルダの `summary/`。毎回作り直す）：

- `pairs.csv`：組（ターゲット，ソース）× K ごとに，終わった反復の数，UAR の平均と 95% 区間
  （ターゲットのみ・ソースのみ・合併），合併 − ターゲットのみ の差の平均と 95% 区間。
  区間は**同じ反復どうしの対の差の t 区間**（`docs/researchvision.md` 6.2）。
  反復ごとの UAR の t 区間は「抽出と分割の揺れ」だけを表す（話者を集め直す揺れは含まない）
- `kstar.csv`：組ごとの K*＝差の 95% 区間の下端が 0 を上回る**最大の** K（`docs/methods.md` 2.4）。
  どの K でも上回らなければ `none`。**反復が 30 回そろっていない組は `provisional`**。
  **反復が 2 回未満の組は区間が作れないので `na`（状態 `single_rep`）**。
  **IEMOCAP がターゲットの組は区間の計算法が未決**（テスト話者の組み合わせが 5・7 通りしかなく，
  30 回は独立でない。`docs/researchvision.md` 7 節）なので `iemocap_interval_undecided` と書く
- `c_rule.csv`：作り方 × K ごとの，C の決まり方（cv / tie / fixed）の割合
- `fig_<ターゲット>.png`：ソースごとの小さな図。横軸 K，縦軸 UAR（誤差棒は 95% 区間）
- `fig_gain.png`：合併 − ターゲットのみ の差（UAR の点．行＝ターゲット，列＝ソース，K ごと）。
  区間が 0 をまたがない升に `*`

オラクル（`uar_oracle`）は `pairs.csv` に別の列で残す（分析専用。手法の性能として報告しない）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
K_ORDER = ["1", "5", "10", "25", "50", "all"]
R_FULL = 30
IEMOCAP_TARGETS = ("IEMOCAP-impro", "IEMOCAP-script")
ORDER = ("IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW", "RAVDESS", "CREMA-D")


def ordered(names) -> list[str]:
    """群を `measure/data.py` の GROUPS_ALL と同じ順に並べる（知らない名前は後ろ）。"""
    names = list(dict.fromkeys(names))
    return [g for g in ORDER if g in names] + sorted(g for g in names if g not in ORDER)

# 色（dataviz の参照パレット。最初の 3 色は全組み合わせで色覚の検査済み）
C_MERGED, C_TGT, C_SRC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "gain", ["#a1282a", "#e34948", "#f0efec", "#3987e5", "#104281"])


def load(dir_: Path) -> pd.DataFrame:
    files = sorted(p for p in dir_.glob("*.csv"))
    if not files:
        raise SystemExit(f"{dir_} に結果の CSV が無い")
    return pd.concat([pd.read_csv(p, comment="#", dtype={"K": str, "source": str},
                                  keep_default_na=False, na_values=[""])
                      for p in files], ignore_index=True)


def t_interval(x: np.ndarray) -> tuple[float, float, float]:
    """平均と 95% の t 区間。2 回未満なら区間は NaN。"""
    x = np.asarray(x, float)
    m = float(x.mean()) if len(x) else np.nan
    if len(x) < 2:
        return m, np.nan, np.nan
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    tgt = df[df.method == "tgt_only"].set_index(["target", "K", "rep"])
    src = df[df.method == "src_only"].set_index(["target", "source", "rep"])
    mer = df[df.method == "merged"]
    rows = []
    for (t, s, K), g in mer.groupby(["target", "source", "K"]):
        reps = g["rep"].to_numpy()
        u_m = g["uar"].to_numpy()
        u_t = tgt.loc[[(t, K, r) for r in reps], "uar"].to_numpy()
        u_s = src.loc[[(t, s, r) for r in reps], "uar"].to_numpy()
        row = {"target": t, "source": s, "K": K, "n_rep": len(reps)}
        for name, u in (("tgt", u_t), ("src", u_s), ("merged", u_m), ("diff", u_m - u_t)):
            row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = t_interval(u)
        row["tgt_oracle_mean"] = float(tgt.loc[[(t, K, r) for r in reps], "uar_oracle"].mean())
        row["merged_oracle_mean"] = float(g["uar_oracle"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["K"] = pd.Categorical(out["K"], K_ORDER, ordered=True)
    return out.sort_values(["target", "source", "K"]).reset_index(drop=True)


def kstar_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, s), g in pairs.groupby(["target", "source"], observed=True):
        n = int(g["n_rep"].min())
        if n < 2:                                         # 区間が作れない（速報の 1 回など）
            rows.append({"target": t, "source": s, "K_star": "na", "n_rep": n,
                         "status": "single_rep"})
            continue
        sig = g[g["diff_lo"] > 0]
        kstar = str(sig["K"].max()) if len(sig) else "none"
        status = ("iemocap_interval_undecided" if t in IEMOCAP_TARGETS
                  else "final" if n >= R_FULL else "provisional")
        rows.append({"target": t, "source": s, "K_star": kstar, "n_rep": n, "status": status})
    return pd.DataFrame(rows)


def c_rule_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.method != "src_only"]
    out = (d.groupby(["method", "K"])["C_rule"].value_counts(normalize=True)
           .unstack(fill_value=0.0).reset_index())
    out["K"] = pd.Categorical(out["K"], K_ORDER, ordered=True)
    return out.sort_values(["method", "K"]).reset_index(drop=True)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def fig_target(pairs: pd.DataFrame, target: str, kind: str, path: Path) -> None:
    g = pairs[pairs.target == target]
    sources = ordered(g["source"])
    ncol = 3
    nrow = int(np.ceil(len(sources) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.1 * nrow), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    x = np.arange(len(K_ORDER))
    cols_lo = ["tgt_lo", "merged_lo", "src_lo", "tgt_mean", "merged_mean", "src_mean"]
    cols_hi = ["tgt_hi", "merged_hi", "src_hi", "tgt_mean", "merged_mean", "src_mean"]
    lo_all = min(float(np.nanmin(g[cols_lo].to_numpy())), 0.25)   # 区間が無ければ平均から
    hi_all = float(np.nanmax(g[cols_hi].to_numpy()))
    for ax, s in zip(axes.flat, sources):
        h = g[g.source == s].set_index("K").reindex(K_ORDER)
        _style(ax)
        ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=":")
        for name, col, lab in (("tgt", C_TGT, "target only"), ("merged", C_MERGED, "merged"),
                               ("src", C_SRC, "source only")):
            m = h[f"{name}_mean"].to_numpy()
            err = np.vstack([m - h[f"{name}_lo"].to_numpy(), h[f"{name}_hi"].to_numpy() - m])
            ls = "--" if name == "src" else "-"
            ax.errorbar(x, m, yerr=np.nan_to_num(err), color=col, linewidth=2, linestyle=ls,
                        marker="o", markersize=5, capsize=2, elinewidth=1, label=lab)
        n = int(h["n_rep"].min())
        ax.set_title(f"source: {s}  (reps {n})", fontsize=9, color=INK, loc="left")
        ax.set_xticks(x, K_ORDER)
        ax.set_ylim(lo_all - 0.02, hi_all + 0.02)
    for ax in axes.flat[len(sources):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("UAR (test speakers)", fontsize=9, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel("K (labelled target clips per emotion)", fontsize=9, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    n_min = int(g["n_rep"].min())
    bars = "bars = 95% CI" if n_min >= 2 else "single rep: no CI"
    fig.suptitle(f"target: {target}   [{kind}]   dotted = chance (0.25), {bars}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_gain(pairs: pd.DataFrame, kind: str, path: Path) -> None:
    groups = ordered(list(pairs["target"]) + list(pairs["source"]))
    vmax = float(np.nanmax(np.abs(pairs["diff_mean"]))) * 100 or 1.0
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=SURFACE)
    for n_ax, (ax, K) in enumerate(zip(axes.flat, K_ORDER)):
        h = pairs[pairs.K == K]
        M = np.full((len(groups), len(groups)), np.nan)
        sig = np.zeros_like(M, bool)
        for _, r in h.iterrows():
            i, j = groups.index(r.target), groups.index(r.source)
            M[i, j] = r.diff_mean * 100
            sig[i, j] = (r.diff_lo > 0) or (r.diff_hi < 0)
        im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color=MUTED)
                elif np.isfinite(M[i, j]):
                    dark = abs(M[i, j]) > 0.55 * vmax           # 濃い升は白い文字
                    ax.text(j, i, f"{M[i, j]:+.1f}{'*' if sig[i, j] else ''}", ha="center",
                            va="center", fontsize=7, color="#ffffff" if dark else INK)
        first_col, bottom = n_ax % 3 == 0, n_ax >= 3
        ax.set_xticks(range(len(groups)), groups if bottom else [], rotation=45, ha="right",
                      fontsize=7)
        ax.set_yticks(range(len(groups)), groups if first_col else [], fontsize=7)
        ax.set_title(f"K = {K}", fontsize=10, color=INK, loc="left")
        if bottom:
            ax.set_xlabel("source", fontsize=8, color=INK2)
        if first_col:
            ax.set_ylabel("target", fontsize=8, color=INK2)
    fig.colorbar(im, ax=axes, shrink=0.6,
                 label="merged − target only (UAR points; * = 95% CI excludes 0)")
    n_min = int(pairs["n_rep"].min())
    note = "" if n_min >= 2 else "  SINGLE REP: no CI, no * marks"
    fig.suptitle(f"Gain from adding the source  [{kind}]  reps {n_min}–{int(pairs['n_rep'].max())}"
                 f"  (blank = not run yet or same group){note}",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def summarize(dir_: Path) -> Path:
    df = load(dir_)
    out = dir_ / "summary"
    out.mkdir(exist_ok=True)
    kind = str(df["kind"].iloc[0])
    pairs = pair_table(df)
    pairs.to_csv(out / "pairs.csv", index=False)
    kstar_table(pairs).to_csv(out / "kstar.csv", index=False)
    c_rule_table(df).to_csv(out / "c_rule.csv", index=False)
    for t in dict.fromkeys(pairs["target"]):
        fig_target(pairs, t, kind, out / f"fig_{t}.png")
    fig_gain(pairs, kind, out / "fig_gain.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="地図の途中結果をまとめる")
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "map_audio_wavlm")
    args = ap.parse_args()
    out = summarize(args.dir)
    n = load(args.dir).groupby("target")["rep"].nunique()
    print("ターゲットごとの終わった反復:", n.to_dict())
    print(f"書いた: {out}")


if __name__ == "__main__":
    sys.exit(main())
