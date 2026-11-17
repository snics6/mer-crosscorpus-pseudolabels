"""実験 20（DAI-20）：提案の手法（感情ごとの信頼度の表で声・顔・言葉の答えをまとめる．水準は K 本，相性はソース）の評価の回．
設計：docs/設計/実験20_設計.md．単位の計算は実験 16 の台本（dai16_probe.run_unit）をそのまま呼び，実験 1 の評価の回の保存に当てる．
この台本が足すのは，比べる相手（確率の平均，確信度で重みづける平均，K 本で合わせたモダリティごとの重み，K 本で重みの表を探す版），
提案の部品を外した版（相性をソースの中の値から取る版），実験 1 の 5 つの比較手法の融合の分類器の判定の写し，である．
足す版は，単位の保存の確率（float32）だけから作る（学び直さない．後から同じ値を作り直せる）．

段：
  selftest   人工の数での検算
  freeze     下調べが選んだ設定（n0・β・戻し先）と比べる相手の設定を frozen/dai20_frozen.json に写す（コミットしてから units を流す）
  units      単位ごとの判定
  summary    UAR と区間，差と区間，格子の全点の UAR と 3 通りの選び方，感情ごとの再現率，仮定の表，実験 1・2 から写す値

使い方：.venv/bin/python exp/dai20_eval.py --stage units [--targets AFEW]
台本の検査：--tag pseudo --out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW --folds 0 --seeds 0
            （仮のターゲットの回に同じ計算を当てる．実験 16 の保存と同じ判定になるはず．評価の回の出力には触れない）
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai20_eval"
DESIGN = ROOT / "docs" / "設計" / "実験20_設計.md"
FROZEN = ROOT / "frozen" / "dai20_frozen.json"
PROBE = {5: ROOT / "results" / "dai16_probe", 25: ROOT / "results" / "dai16_probe", 10: ROOT / "results" / "dai16_probe_k10"}
B_SRC = ROOT / "results" / "dai1" / "stats" / "desc" / "levels.csv"
C_SRC = ROOT / "results" / "dai2" / "stats" / "prop_minus_method.csv"
CORP = tuple(DD.CORPORA)
MODS = "AVT"
KS = (5, 10, 25)
NF = P16.NF
GRID = P16.GRID
NG = len(GRID)
SEED = 20261114 + 20
WF_MS = (10, 100, 1000)                  # K 本で重みの表を探す版：乱数の表の数（等しい重みの表を別に 1 つ足す）
N_WF = max(WF_MS)
TAU_T = P16.TAU_F8                       # K 本で合わせたモダリティごとの重み：1 へ引く罰則の強さ（実験 16 の F8 と同じ 0.01．この相手のために選んでいない）
A_VERS = ("prod", "psel", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb") + tuple(f"wf{m}" for m in WF_MS)   # 比べる相手 A（同じ 3 つの分類器の答えのまとめ方）
ABL = ("prop_noprof", "prop_noprof_sel", "prop_noint", "prop_n0zero", "prop_home", "prop_levelfree")             # 提案の部品を 1 つ外した版
TEMP_MODES = {"tcal": "sep", "temp": "joint", "tempb": "joint_b"}
REF_VERS = ("M_F6_ext", "M_F8")                                        # 参考（ソースのラベルだけで作る表）
ORC_VERS = ("M_F6_orc",)                                               # 参照（T の候補の全ラベルで測った表．実際には使えない）
B_METHODS = ("kd", "agree", "mix", "cotrain", "coreg")                 # 実験 1 の 5 つの比較手法（融合の分類器．主の版）
B_VERS = tuple(f"b_{m}" for m in B_METHODS)
PLAIN = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T") + B_VERS                # 切片をずらさない判定だけを持つ版（学び直した 1 つの分類器）
B_ROWS = {"none": "none_AVT", "joint|one|main": "joint_AVT", "cand:kd|one": "b_kd", "cand:agree|one": "b_agree", "cand:mix|one": "b_mix",
          "cand:cotrain|ct": "b_cotrain", "cand:coreg|cr": "b_coreg"}   # 実験 1 の集計の行の名前 → この台本の版
SEED_SETS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10)), "0-2": (0, 1, 2)}
SCRIPTS = ("dai20_eval.py", "dai16_probe.py", "dai15.py", "dai14.py", "dai12_probe.py", "dai11_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py",
           "dai_data.py")                                                  # この台本が読み込む exp の台本の全部（dai14・dai11_probe は読み込まれるだけで，計算の道では呼ばれない）
RUN = {"commit": "", "dirty": True}
WATCH = tuple(f"exp/{n}" for n in SCRIPTS) + ("frozen/dai20_frozen.json", "splits", "measure", "features", "data/metadata", "requirements.txt",
                                                  "docs/設計/実験20_設計.md")


def same_code(commit: str, other: str = "HEAD") -> bool:
    """2 つのコミットのあいだで，計算に関わるファイル（台本と部品，固定のファイル，分割，読み込み，発話一覧，設計）に差が無いか．
    単位の段：済みの単位の csv のコミットと今の HEAD を比べ，差が無いときだけ飛ばす（関係の無いコミットの後でも続きから流せる．差があれば止める）．
    集計の段：単位どうしを比べる（全部の単位が同じ台本と入力で作られたことだけを確かめる．集計の段だけを後から直して流し直せる）．"""
    if str(commit) == str(other) or (other == "HEAD" and commit == RUN["commit"]):
        return True
    return subprocess.run(["git", "diff", "--quiet", str(commit), str(other), "--", *WATCH], cwd=ROOT).returncode == 0
WS = np.concatenate([np.ones((1, 3, 4)), 3.0 * np.random.default_rng([SEED, 1]).dirichlet(np.ones(3), size=(N_WF, 4)).transpose(0, 2, 1)])
# WS：(1 + N_WF, 分類器 3, 感情 4)．表 0 は等しい重み（全部 1）．表 1〜N_WF は，感情ごとに 3 つの分類器の重みを Dirichlet(1, 1, 1) から引いて 3 倍した表
# （感情ごとの和は 3，12 個の平均は 1．提案の重みと同じ大きさ）．全部の単位と K で同じ表を使う．候補の数 m の版は，表 0 と表 1〜m を候補にする．
SETTINGS = dict(KS=list(KS), layer="adj（候補の上の確率の平均を積に合わせる切片のずらし．ラベルを使わない）",
                wf=dict(tables="Dirichlet(1,1,1) を感情ごとに引いて 3 倍．表 0 は全部 1", seed=[SEED, 1], ms=list(WF_MS),
                        select="K 本の上の UAR（組 j の発話は，組 j を抜いて作った分類器の確率と，その分類器の候補の上の確率から決めた切片で判定）が最大の表",
                        tie="最大に並んだ表の平均を使う（最大が表 0 だけなら積の判定をそのまま使う）"),
                temp=dict(fit="K 本の抜いた確率（中心化した対数）の，感情ごとに等しい重みの対数尤度", tau=TAU_T, bounds=[0.0, P16.W_MAX],
                          tcal="分類器ごとに別々に，その分類器だけの対数尤度で重み 1 つを合わせる", temp="3 つの重みを，まとめた答えの対数尤度で同時に合わせる",
                          tempb="切片 4 つも同時に合わせ，重みだけを使う"),
                wf_tables_sha256=hashlib.sha256(WS.tobytes()).hexdigest(),
                conf="発話ごとに，各分類器の重み ＝ その分類器の最大の確率", avg="3 つの確率の平均",
                psel="実験 1 が別のコーパスで選んだ指数（固定のファイルの w）を平均 1 に割って掛ける積")


def dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def same_but(i: int, n0=None, beta=None) -> int:
    """格子の点 i と同じ設定で，n0 か β だけを替えた点の番号．"""
    a, b, p = GRID[i]
    return GRID.index((a if n0 is None else n0, b if beta is None else beta, p))


def seed_rng(*key) -> np.random.Generator:
    return P16.row_rng("dai20", *key)


# ------------------------------------------------------------------ 固定のファイル

def probe_choice(K: int, c: str, betas: tuple) -> tuple[int, int, set]:
    """下調べの単位の csv から，コーパス c の点を選び直す：c を評価のターゲットにした単位（c が仮のターゲットにもソースにも入らない）の
    adj の層の UAR の平均が最大の点（同点は番号の小さい点．実験 16 の集計と同じ規則）．戻り：(点, 単位の数, その単位に現れたコーパス)．"""
    ud = PROBE[K] / "units"
    rows, seen, n = [], set(), 0
    for p in sorted(ud.glob(f"{c}_*.csv")):
        d = pd.read_csv(p, low_memory=False)
        d = d[(d.K == K) & d.version.astype(str).str.fullmatch(r"M_F6_mix\d+")]
        if not len(d):
            continue
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        seen |= set(map(str, z["srcs"])) | set(d.c.astype(str))
        if not (d["T"] == c).all():
            raise SystemExit(f"{p.name}：T が {c} でない")
        rows.append(d[["version", "uar_adj"]])
        n += 1
    g = pd.concat(rows).groupby("version").uar_adj.mean()
    vals = [float(g[f"M_F6_mix{i}"]) if GRID[i][1] in betas else -np.inf for i in range(NG)]
    return int(np.argmax(vals)), n, seen


def stage_freeze() -> None:
    choice, shas, checks = {}, {}, []
    for K in KS:
        src = PROBE[K] / "summary" / "mixsel.csv"
        ms = pd.read_csv(src)
        ms = ms[(ms.version == "M_F6_mixsel") & (ms.K == K)]
        if sorted(ms.c) != sorted(CORP):
            raise SystemExit(f"下調べの選んだ点が 4 コーパスそろっていない（K={K}）")
        for r in ms.itertuples():                                           # mixsel.csv の c は「評価するコーパス」＝この実験の評価のターゲット T
            i = int(str(r.chosen).replace("M_F6_mix", ""))
            assert f"M_F6_mix{i}" == r.chosen and (float(r.n0), float(r.beta), str(r.prior)) == (GRID[i][0], GRID[i][1], GRID[i][2]), (r.c, K, r.chosen)
            i2, n, seen = probe_choice(K, r.c, (0.0, 1.0))
            if i2 != i or r.c in seen or n != 30:
                raise SystemExit(f"選び直しが合わない：{r.c} K={K} 保存 {i} 選び直し {i2} 単位 {n} 現れたコーパス {sorted(seen)}")
            j, _, _ = probe_choice(K, r.c, (0.0,))                             # 相性を使わない 8 点の中から同じ規則で選んだ点
            choice.setdefault(r.c, {})[str(K)] = dict(grid=i, n0=("inf" if np.isinf(GRID[i][0]) else GRID[i][0]), beta=GRID[i][1], prior=GRID[i][2],
                                                      noprof_sel=j)
            checks.append(dict(T=r.c, K=K, grid=i, n_units=n, corpora_in_units=sorted(seen), noprof_sel=j))
        shas[str(src.relative_to(ROOT))] = dai15.sha(src)
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} が既にある（作り直すなら消してから）")
    ref = {str(q.relative_to(ROOT)): dai15.sha(q) for q in sorted({PROBE[K] / "summary" / "reliability_src.csv" for K in KS}) if q.exists()}
    json.dump(dict(choice=choice, source_sha256=shas, reference_sha256=ref, selfcheck=checks, settings=SETTINGS,
                   rule="下調べ（実験 16 と K = 10 の追補）の M_F6_mixsel：評価するコーパスを含まない 30 単位の adj の UAR の平均が最大の格子の点",
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=RUN["commit"]), open(FROZEN, "w"), ensure_ascii=False, indent=1)
    print(f"freeze：{FROZEN}", flush=True)


def load_frozen() -> dict:
    fz = json.load(open(FROZEN))
    for p, h in fz["source_sha256"].items():
        if (ROOT / p).exists() and dai15.sha(ROOT / p) != h:
            raise SystemExit(f"固定のファイルの元（{p}）が変わっている")
    if json.dumps(fz["settings"], ensure_ascii=False, sort_keys=True) != json.dumps(SETTINGS, ensure_ascii=False, sort_keys=True):
        raise SystemExit("固定のファイルの比べる相手の設定が台本と違う")
    return {c: {int(K): (int(v["grid"]), int(v["noprof_sel"])) for K, v in d.items()} for c, d in fz["choice"].items()}


# ------------------------------------------------------------------ 単位

def unit_list(tag: str, targets, folds, seeds) -> list:
    """(T, c, f, r, 名前)．評価の回は c = T（ソースは T 以外の 3 コーパス）．"""
    fz1, _ = dai3.load_fz1(OUT1)
    out = []
    for T in (targets or CORP):
        if tag == "eval":
            todo = [(T, f, r) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))]
        else:
            todo = dai3.pseudo_units(OUT1, T)
        for c, f, r in todo:
            if (folds is None or f in folds) and (seeds is None or r in seeds):
                out.append((T, c, f, r, f"{T}_{'ALL' if tag == 'eval' else c}_f{f}_r{r}"))
    return out


def match_shift_batch(Lp: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """P16.match_shift を表の束 Lp (表, 発話, 4) に当てる．表ごとに同じ式・同じ止め方（収束した表は動かさない）．
    戻り：(切片のずれ (表, 4)，反復の上限までに収束しなかった表の数)．"""
    d = torch.zeros(Lp.shape[0], 4, dtype=Lp.dtype, device=Lp.device)
    lt = torch.log(target.clamp(min=1e-12))
    live = torch.arange(Lp.shape[0], device=Lp.device)
    for _ in range(P16.ADJ_ITERS):
        if not len(live):
            break
        m = torch.softmax(Lp[live] + d[live, None, :], -1).mean(1)
        done = (m - target).abs().amax(1) < P16.ADJ_TOL * 1e-2
        upd = d[live] + lt - torch.log(m.clamp(min=1e-12))
        upd = upd - upd.mean(1, keepdim=True)
        d[live[~done]] = upd[~done]
        live = live[~done]
    return d, int(len(live))


def wf_search(LMh: np.ndarray, yk: np.ndarray, fold: np.ndarray, LMpj: list) -> tuple[np.ndarray, int]:
    """表 WS のそれぞれについて，K 本の上の UAR（%）を adj の層で測る．組 j の発話は，組 j を抜いて作った分類器の点（LMh の行）に，
    その分類器の候補の上の点（LMpj[j]：(3, 候補, 4)）からラベルを使わずに決めた切片を足して判定する（実験 16 の入れ子と同じ切片の決め方）．"""
    dv = dev()
    W = torch.tensor(WS, dtype=torch.float64, device=dv)
    pred = torch.empty((len(WS), len(yk)), dtype=torch.long, device=dv)
    ncap = 0
    for j in range(NF):
        te = np.flatnonzero(fold == j)
        Lj = torch.tensor(LMpj[j], dtype=torch.float64, device=dv)
        tgt = torch.softmax(Lj.sum(0), -1).mean(0)
        step = max(1, int(2.5e7 // (Lj.shape[1] * 4)))                    # 1 回に載せる表の数（GPU のメモリを抑える）
        Lh = torch.tensor(LMh[:, te], dtype=torch.float64, device=dv)
        for a in range(0, len(WS), step):
            Sp = torch.einsum("rky,kny->rny", W[a:a + step], Lj)
            Sp = Sp - torch.logsumexp(Sp, -1, keepdim=True)
            d, nc = match_shift_batch(Sp, tgt)
            ncap += nc
            Sh = torch.einsum("rky,kny->rny", W[a:a + step], Lh)
            Sh = Sh - torch.logsumexp(Sh, -1, keepdim=True)
            pred[a:a + step, te] = (Sh + d[:, None, :]).argmax(-1)
            del Sp
    y = torch.tensor(yk, device=dv)
    hit = (pred == y[None]).double()
    ua = torch.stack([hit[:, y == c].mean(1) for c in range(4) if bool((y == c).any())], 1).mean(1)
    return (100.0 * ua).cpu().numpy(), ncap


def wf_pick(ua: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """候補（表 0 と表 1〜m）のうち K 本の上の UAR が最大の表．最大に並んだ表は平均する．戻り：(重みの表 (3, 4), 並んだ表の番号)．"""
    u = ua[:m + 1]
    tied = np.flatnonzero(u >= u.max() - 1e-9)
    return WS[tied].mean(0), tied


def temp_w(LM: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    """モダリティごとの重み（温度の逆数に当たる）を，感情ごとに等しい重みの対数尤度で合わせる（1 へ引く罰則 TAU_T，範囲 0〜W_MAX）．
    sep：分類器ごとに別々に，その分類器だけの答えの対数尤度で合わせる（分類器ごとの温度の較正）．
    joint：3 つを同時に，まとめた答えの対数尤度で合わせる（重みつき対数線形プール）．joint_b：joint に切片 4 つを足して合わせ，重みだけを使う．"""
    if mode == "sep":
        return np.array([float(P16.fit_wb(LM[k:k + 1], y, np.ones(1), TAU_T, fit_w=True, fit_b=False)[0][0]) for k in range(LM.shape[0])])
    return np.asarray(P16.fit_wb(LM, y, np.ones(LM.shape[0]), TAU_T, fit_w=True, fit_b=(mode == "joint_b"))[0], float)


def b_preds(sv, K: int) -> dict:
    """実験 1 の 5 つの比較手法の融合の分類器（主の版 one．実験 1 が仮のターゲットで選んで固定した設定）のテストの判定を，実験 1 の保存から写す．"""
    out = {}
    for pm in ("kd", "agree", "mix"):
        pp = {k: v for k, v in sv.fz["select"][f"AVT|{K}|{pm}|one"].items() if k != "pseudo_uar"}
        out[f"b_{pm}"] = sv.p3[f"p3_AVT_K{K}_pred"][sv.p3_index("AVT", K, method=pm, teacher="one", **pp)]
    rho = sv.fz["select"][f"AVT|{K}|cotrain|ct"]["rho"]
    out["b_cotrain"] = sv.p3[f"p3_AVT_K{K}_pred"][sv.p3_index("AVT", K, method="cotrain", teacher="ct", rho=rho)]
    rho = sv.fz["select"][f"AVT|{K}|coreg|cr"]["rho"]
    out["b_coreg"] = sv.p3[f"p3g_K{K}_AVT_pred"][list(dai1.COREG_RHO).index(float(rho))]
    return out


def add_extra(arr: dict, rows: list, bp: dict, wsel: dict) -> None:
    """実験 16 の単位の出力（arr・rows）に，比べる相手と部品を外した版の判定を足す．使うのは arr の確率（float32）と表だけ．
    bp：{K: {版: 判定}}（実験 1 の比較手法の判定の写し）．wsel：{K: 指数 3 つ}（実験 1 が別のコーパスで選んだ積の指数）．"""
    yt, yp = arr["y_test"].astype(int), arr["y_pool"].astype(int)
    pairs = list(itertools.combinations(range(NF), 2))
    pidx = {p: n for n, p in enumerate(pairs)}
    for K in KS:
        kpos, fold = arr[f"kpos_K{K}"], arr[f"kfold_K{K}"].astype(int)
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(len(yp)), kpos)
        Pt = [arr[f"p_test_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
        Pp = [arr[f"p_pool_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
        Ph = [arr[f"pcf_{m}_K{K}"].astype(np.float64) for m in MODS]           # K 本の発話に，その発話の組を抜いて作り直した分類器が付けた確率
        P2 = [arr[f"pcf2_{m}_K{K}"].astype(np.float64) for m in MODS]         # 2 組を抜いて作り直した分類器の，K 本の上の確率 (10, 4K, 4)
        PJ = [arr[f"ppj_{m}_K{K}"].astype(np.float64) for m in MODS]          # 組 j を抜いて作り直した分類器の，候補の上の確率 (5, 候補, 4)
        LMt, LMp, LMh = (np.stack([P16.centered(p) for p in Ps]) for Ps in (Pt, Pp, Ph))
        LMpj = [np.stack([P16.centered(PJ[k][j]) for k in range(3)]) for j in range(NF)]
        target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)                    # 積の，候補の上の確率の平均

        def put(v: str, St: np.ndarray, Sp: np.ndarray, **extra) -> np.ndarray:
            dl, rs = P16.match_shift(Sp, target)
            pr, pj, pp = St.argmax(1), (St + dl).argmax(1), (Sp + dl)[nk].argmax(1)
            arr[f"pred_raw_K{K}_{v}"], arr[f"pred_adj_K{K}_{v}"], arr[f"pred_pool_K{K}_{v}"] = pr.astype(np.int8), pj.astype(np.int8), pp.astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pj), uar_pool=100 * S.uar(yp[nk], pp), adj_resid=rs, **extra))
            return pj

        def f6(W: np.ndarray, LM: np.ndarray) -> np.ndarray:
            return P16.norm_log(np.einsum("ky,kny->ny", W, LM))

        # ---- 別のコーパスで選んだ指数の積（指数は平均 1 に割る．切片をずらす前の判定は実験 1 の prod_sel と同じ式）
        ws = np.asarray(wsel[K], float)
        ws = ws / ws.mean() if ws.mean() > 0 else np.ones(3)
        put("psel", P16.norm_log(np.tensordot(ws, LMt, axes=1)), P16.norm_log(np.tensordot(ws, LMp, axes=1)), w_A=ws[0], w_V=ws[1], w_T=ws[2])
        # ---- 確率の平均，確信度で重みづける平均
        put("avg", P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0)))
        ct, cp = np.stack([p.max(1) for p in Pt]), np.stack([p.max(1) for p in Pp])
        put("conf", P16.lg((ct[:, :, None] * np.stack(Pt)).sum(0) / ct.sum(0)[:, None]), P16.lg((cp[:, :, None] * np.stack(Pp)).sum(0) / cp.sum(0)[:, None]))
        # ---- K 本で合わせたモダリティごとの重み（tcal：分類器ごとに別々．temp：3 つを同時．tempb：切片も同時に合わせ，重みだけを使う）
        for v, fb in TEMP_MODES.items():
            w = temp_w(LMh, yk, fb)
            est = np.empty((len(kpos), 4))                                      # K 本での見積もり（入れ子）：組 j の発話は，組 j を使わずに合わせた重みで当てる
            for j in range(NF):
                te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
                Ltr = np.stack([P16.centered(np.stack([P2[k][pidx[tuple(sorted((int(fold[a]), j)))]][a] for a in tr])) for k in range(3)])
                wj = temp_w(Ltr, yk[tr], fb)
                dl, _ = P16.match_shift(P16.norm_log(np.tensordot(wj, LMpj[j], axes=1)), np.exp(P16.norm_log(LMpj[j].sum(0))).mean(0))
                est[te] = P16.norm_log(np.tensordot(wj, LMh[:, te], axes=1) + dl)
            put(v, P16.norm_log(np.tensordot(w, LMt, axes=1)), P16.norm_log(np.tensordot(w, LMp, axes=1)), w_A=w[0], w_V=w[1], w_T=w[2],
                w_bound=float(((w <= 1e-9) | (w >= P16.W_MAX - 1e-9)).mean()), est_uar=100 * S.uar(yk, est.argmax(1)), est_ll=P16.bal_ll(est, yk))
            arr[f"W_{v}_K{K}"] = w
        # ---- K 本で重みの表を探す版（候補の数 m ごと）
        ua, ncap = wf_search(LMh, yk, fold, LMpj)
        arr[f"wf_uar_K{K}"] = ua.astype(np.float32)
        for m in WF_MS:
            W, tied = wf_pick(ua, m)
            ex = dict(wf_index=int(tied[0]), wf_n_tied=len(tied), wf_uar_k=float(ua[tied[0]]), wf_uar_k_eq=float(ua[0]), wf_n_capped=ncap)
            arr[f"wf_tied{m}_K{K}"] = tied.astype(np.int16)                      # 最大に並んだ表の番号
            if len(tied) == 1 and tied[0] == 0:                                  # 等しい重みの表だけが最大：積の判定をそのまま使う
                for w_ in ("raw", "adj"):
                    arr[f"pred_{w_}_K{K}_wf{m}"] = arr[f"pred_raw_K{K}_prod"]
                arr[f"pred_pool_K{K}_wf{m}"] = arr[f"pred_pool_K{K}_prod"]
                u_ = 100 * S.uar(yt, arr[f"pred_raw_K{K}_prod"].astype(int))
                rows.append(dict(K=K, version=f"wf{m}", uar_raw=u_, uar_adj=u_, uar_pool=100 * S.uar(yp[nk], arr[f"pred_pool_K{K}_prod"].astype(int)), adj_resid=0.0, **ex))
            else:
                put(f"wf{m}", f6(W, LMt), f6(W, LMp), **ex)
            arr[f"W_wf{m}_K{K}"] = W
        # ---- 相性をソースの中の値（home．残り方を掛けない）から取る版．あわせて，保存の確率から提案の点を作り直して実験 16 の判定と照らす
        DK, Dext, Dhome = arr[f"D_M_K_K{K}"], arr["SR_M_ext"], arr["SR_M_home"]
        agree = []
        for i, (n0, beta, prior) in enumerate(GRID):
            if beta != 1.0:
                continue
            We = P16.to_w(P16.mix_rel(DK, Dext, 4 * K, n0, 1.0, prior))
            dl, _ = P16.match_shift(f6(We, LMp), target)
            agree.append(float(((f6(We, LMt) + dl).argmax(1) == arr[f"pred_adj_K{K}_M_F6_mix{i}"]).mean()))
            lev = P16.mix_rel(DK, Dext, 4 * K, n0, 0.0, prior)                  # 水準だけ（提案と同じ n0・戻し先）
            Wh = P16.to_w(np.maximum(0.0, lev + (Dhome - Dhome.mean(1, keepdims=True))))
            put(f"M_F6h_mix{i}", f6(Wh, LMt), f6(Wh, LMp))
        e0 = float((f6(np.ones((3, 4)), LMt).argmax(1) == arr[f"pred_raw_K{K}_prod"]).mean())
        e1 = float((arr[f"pred_raw_K{K}_psel"] == arr[f"pred_raw_K{K}_prod_sel"]).mean())
        rows.append(dict(K=K, version="_check20", recheck_agree_min=min(agree), eq_agree=e0, psel_agree=e1))
        # ---- 実験 1 の比較手法の判定の写し
        for v, p in bp[K].items():
            if len(p) != len(yt):
                raise SystemExit(f"{v} の判定の数がテストと違う（K={K}）")
            arr[f"pred_raw_K{K}_{v}"] = np.asarray(p).astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, np.asarray(p).astype(int))))


def stage_units(out: Path, tag: str, targets, folds, seeds) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    load_frozen()                                                           # 固定のファイルの設定が台本と合うことを，計算の前に確かめる
    udir = out / "units"
    udir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for T, c, f, r, name in unit_list(tag, targets, folds, seeds):
        if (udir / f"{name}.csv").exists():
            if all(same_code(x) for x in pd.read_csv(udir / f"{name}.csv", low_memory=False).commit.unique()):
                print(f"{name}：済み（飛ばす）", flush=True)
                continue
            raise SystemExit(f"{name} が，計算に関わるファイルの違うコミットで既にある")
        t0 = time.time()
        sv = dai3.Saved(OUT1, tag, T, c, f, r, fz1[T])
        rows, arr = P16.run_unit(sv)
        t1 = time.time()
        add_extra(arr, rows, {K: b_preds(sv, K) for K in KS}, sv.wsel)
        np.savez_compressed(udir / f"{name}.npz", **arr)                    # npz を先に書き，csv を済みの印にする
        df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, tag=tag, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0, sec_extra=time.time() - t1,
                                       host=platform.node(), gpu=gpu, dai1_commits=",".join(sorted(sv.commits)))
        df.to_csv(udir / f"{name}.csv", index=False)
        ck, c2 = df[df.version == "_check"], df[df.version == "_check20"]
        print(f"{name}：{time.time() - t0:.0f} 秒（足した版 {time.time() - t1:.0f} 秒）．行 {len(df)}．作り直しの差 {ck.teacher_dev.max():.1e}．勾配の最大 {ck.grad.max():.1e}．"
              f"作り直した提案の判定の一致 {c2.recheck_agree_min.min():.4f}", flush=True)
        del sv, rows, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ 集計

def role_of(v: str) -> str:
    b = v.split("@")[0]
    return ("proposal" if b == "prop" else "A" if b in A_VERS else "ablation" if b in ABL else "selection" if b == "prop_loto" else
            "oracle" if b in ORC_VERS + ("oracle_grid",) else "reference" if b in REF_VERS else "single" if b.startswith("ret_") else "B")


def stage_summary(out: Path, tag: str, allow_partial: bool) -> None:
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    frozen = load_frozen()
    exp = unit_list(tag, None, None, None)
    files = sorted(p.with_suffix(".npz") for p in (out / "units").glob("*.csv"))
    if len(files) != len(exp) and not allow_partial:
        raise SystemExit(f"units の出力が {len(files)} 個（{len(exp)} のはず）")
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv"), low_memory=False) for p in files], ignore_index=True)
    ucom = sorted(map(str, csvs.commit.unique()))
    if not all(same_code(x, ucom[0]) for x in ucom) and not allow_partial:
        raise SystemExit("units の中に，計算に関わるファイルの違うコミットで作った単位が混ざっている")
    if csvs.duplicated(["T", "c", "fold", "seed", "K", "version"]).any():
        raise SystemExit("units の行が重なっている")
    geff = {c: ST.g_eff_of(c)[0] for c in CORP}
    key = "T" if tag == "eval" else "c"                                    # 成績をまとめるコーパス（評価の回はターゲット T，検査用の回は仮のターゲット c）

    def eff(A, B, c, k_):
        if set(A) != set(B):
            raise SystemExit(f"差の単位がそろわない：{k_}")
        es = ST.effect_stats(A, B, geff[c], seed_rng(*k_))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    n_folds=es["n_folds"], folds_dropped=es.get("folds_dropped", ""), v_cluster=es.get("v_jk", np.nan), v_seed=es.get("v_seed", np.nan),
                    v_seed_all=es.get("v_seed_all", np.nan),                 # jackknife のコーパスでは，外れた分割の種の揺れが v_cluster の側に入る（v_seed_all は全分割の種の項）
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))   # 表の読みやすさのための印（判定ではない）

    # ---- 格子の全点の UAR（adj）と 3 通りの選び方（固定：下調べが選んだ点．loto：ほかの 3 つの評価のターゲットの平均が最大の点．oracle：自分のテストで最大の点）
    g = csvs[csvs.version.astype(str).str.fullmatch(r"M_F6_mix\d+")].copy()
    g["grid"] = g.version.str.replace("M_F6_mix", "").astype(int)
    gl = g.groupby([key, "K", "grid"]).uar_adj.mean().reset_index().rename(columns={key: "c"})
    pick, grows = {}, []
    for c in CORP:
        for K in KS:
            own = gl[(gl.c == c) & (gl.K == K)].set_index("grid").uar_adj
            oth = gl[(gl.c != c) & (gl.K == K)].groupby("grid").uar_adj.mean()      # ほかのコーパスの平均（コーパス 1 つを 1 票）
            if not len(own):
                continue
            i_fz, j_np = frozen[c][K]
            i_lo = int(oth.idxmax()) if len(oth) else i_fz                          # 同点は番号の小さい点
            i_or = int(own.idxmax())
            pick[(c, K)] = dict(prop=i_fz, prop_noprof=same_but(i_fz, beta=0.0), prop_noprof_sel=j_np, prop_n0zero=same_but(i_fz, n0=0.0), prop_loto=i_lo, oracle_grid=i_or,
                                prop_levelfree=GRID.index((float("inf"), 1.0, "eq")))        # 分類器どうしの水準の違いを外した版（3 つの水準を K 本で測った水準の平均にそろえる．相性はソース）
            for i in range(NG):
                grows.append(dict(c=c, K=K, grid=i, n0=GRID[i][0], beta=GRID[i][1], prior=GRID[i][2], uar_adj=float(own.get(i, np.nan)),
                                  uar_adj_others=float(oth.get(i, np.nan)), frozen=bool(i == i_fz), loto=bool(i == i_lo), oracle=bool(i == i_or)))
    pd.DataFrame(grows).to_csv(sdir / "grid_levels.csv", index=False)
    # ---- 塊ごとの数
    cells, emo, ret_emo, ceil, wacc, conf = {}, {}, {}, [], {}, {}
    prof = lambda X: X - X.mean(1, keepdims=True)  # noqa: E731
    cap = math.sqrt(2.0) * float(scipy.stats.norm.ppf(P16.AUC_CLIP[1])) - 1e-9
    for p in files:
        z = np.load(p, allow_pickle=False)
        row0 = csvs[(csvs["T"] + "_" + np.where(csvs.tag == "eval", "ALL", csvs.c) + "_f" + csvs.fold.astype(str) + "_r" + csvs.seed.astype(str)) == p.stem].iloc[0]
        c, f, r = row0[key], int(row0.fold), int(row0.seed)
        yt = z["y_test"].astype(int)
        ret_emo.setdefault((row0["T"], row0["c"]), (z["SR_ret_emo"], z["SR_ret"], z["SR_M_ext"], z["SR_M_home"]))
        for K in KS:
            if (c, K) not in pick:
                continue
            i = pick[(c, K)]["prop"]
            Dk, Do, De = z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["SR_M_ext"]
            pe, po = prof(De), prof(Do)                                         # 相性（行の中の差）：ソースで測った表と，T の候補の全ラベルで測った表
            ceil.append(dict(c=c, K=K, dk_ceiling_share=float((Dk >= cap).mean()), dk_zero_share=float((Dk <= 0).mean()),
                             **{f"profcorr_row_{m}": P16.corr(pe[j], po[j]) for j, m in enumerate(MODS)}, profcorr_noV=P16.corr(pe[[0, 2]], po[[0, 2]])))
            wacc.setdefault((c, K), []).append(np.stack([P16.to_w(P16.mix_rel(Dk, De, 4 * K, *GRID[i])), P16.to_w(Do), P16.to_w(Dk), P16.to_w(De)]))
            for v, kk in (("prop", f"pred_adj_K{K}_M_F6_mix{i}"), ("prod", f"pred_raw_K{K}_prod")):
                np.add.at(conf.setdefault((c, K, v), np.zeros((4, 4))), (yt, z[kk].astype(int)), 1)
            names = {v: f"pred_adj_K{K}_{v}" for v in A_VERS + REF_VERS + ORC_VERS}
            names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in A_VERS})
            names.update({v: f"pred_adj_K{K}_M_F6_mix{j}" for v, j in pick[(c, K)].items()})
            names["prop@raw"] = f"pred_raw_K{K}_M_F6_mix{i}"
            names["prop_noint"] = f"pred_adj_K{K}_M_F6n_mix{i}"                # 提案の重みから「分類器 × 感情」の成分を消した対照（実験 16 の M_F6n）
            names["prop_home"] = f"pred_adj_K{K}_M_F6h_mix{i}" if GRID[i][1] == 1.0 else f"pred_adj_K{K}_M_F6_mix{i}"
            names.update({v: f"pred_raw_K{K}_{v}" for v in PLAIN})
            for v, kk in names.items():
                P12._add(cells.setdefault((c, K, v), {}), (f, r), ST.counts(yt, z[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
    for (c, K, v), dd in cells.items():
        emo[(c, K, v)] = (sum(x[1].sum(0) for x in dd.values()), sum(x[2].sum(0) for x in dd.values()))
    P12._finish(cells)
    # 同じ数になる版：(T, K) ごとに，全部の単位で塊ごと・感情ごとの正解の数が同じ版をまとめ，並びで最初の版の名前を same_as に書く（1 つだけの版は空）
    same_as = {}
    for c in CORP:
        for K in KS:
            vs = [v for (c2, K2, v) in cells if c2 == c and K2 == K]
            vs = [v for v in ("prod", "prop") if v in vs] + [v for v in vs if v not in ("prod", "prop")]
            first, size = {}, {}
            for v in vs:
                dd = cells[(c, K, v)]
                sig = hashlib.sha256(b"".join(repr(k_).encode() + np.ascontiguousarray(dd[k_][1]).tobytes() for k_ in sorted(dd))).hexdigest()
                first.setdefault(sig, v)
                size[sig] = size.get(sig, 0) + 1
                same_as[(c, K, v)] = sig
            for v in vs:
                sig = same_as[(c, K, v)]
                same_as[(c, K, v)] = first[sig] if size[sig] > 1 else ""

    def zero(A: dict) -> dict:                                              # 同じ件数で正解 0 の数（水準の区間を effect_stats で作るための相手．実験 1 の dai1_desc.zero_of と同じ）
        return {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}

    def only(A: dict, i: int) -> dict:                                      # 感情 i の列だけ残した数（再現率とその区間は，得た値を 4 倍する．dai1_desc.emo_of と同じ）
        m = np.zeros(4); m[i] = 1.0
        return {k_: (x[0], x[1] * m, x[2]) for k_, x in A.items()}

    def sub(A: dict, seeds) -> dict:
        return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}

    lrows, erows = [], []
    for (c, K, v), dd in sorted(cells.items()):
        for sn in ("all", "0-2"):
            ds = sub(dd, SEED_SETS[sn])
            if not ds or (sn != "all" and len(ds) == len(dd)):
                continue
            e_ = eff(ds, zero(ds), c, ("L", c, K, v, sn))
            lrows.append(dict(c=c, K=K, version=v, role=role_of(v), seeds=sn, uar=e_["est"], lo=e_["lo"], hi=e_["hi"], se=e_["se"], df=e_["df"], method=e_["method"],
                              n_fr=e_["n_fr"], grid=pick[(c, K)].get(v.split("@")[0], -1), same_as=same_as[(c, K, v)]))
        cor, tot = emo[(c, K, v)]
        full = all((x[2].sum(0) > 0).all() for x in dd.values())             # 全部の単位のテストに 4 感情がそろうときだけ区間を作る
        for i in range(4):
            r_ = eff(only(dd, i), zero(dd), c, ("E", c, K, v, i)) if full else None
            erows.append(dict(c=c, K=K, version=v, role=role_of(v), emotion=i, recall=(4 * r_["est"] if full else np.nan), lo=(4 * r_["lo"] if full else np.nan),
                              hi=(4 * r_["hi"] if full else np.nan), n_fr=(r_["n_fr"] if full else 0), recall_pooled=100 * cor[i] / max(tot[i], 1), n=float(tot[i]),
                              n_units=len(dd), n_per_seed=float(tot[i]) * len({k_[0] for k_ in dd}) / len(dd)))   # n は全部の単位の合計．n_per_seed は種 1 つぶん（別々の発話の数）
    pd.DataFrame(erows).to_csv(sdir / "per_emotion.csv", index=False)
    lv = pd.DataFrame(lrows)
    lv.to_csv(sdir / "levels.csv", index=False)
    con = ([("prop", b) for b in A_VERS + ABL + ("none_AVT", "joint_AVT") + B_VERS + REF_VERS]
           + [(a, "prod") for a in A_VERS[1:] + ("prop_noprof", "prop_noprof_sel", "prop_loto", "oracle_grid") + ORC_VERS + REF_VERS]
           + [("prop_loto", "prop"), ("oracle_grid", "prop"), ("prop", "prop@raw"), ("prop_levelfree", "prod"), ("prop_noint", "prop_noprof")]
           + [("prop@raw", f"{b}@raw") for b in A_VERS] + [("prop", f"{b}@raw") for b in A_VERS[1:]])
    effs = []
    for a, b in con:
        for c in CORP:
            for K in KS:
                A, B = cells.get((c, K, a)), cells.get((c, K, b))
                if A is None or B is None:
                    if allow_partial:
                        continue
                    raise SystemExit(f"差の相手が無い：{(a, b, c, K)}")
                same = all((A[k_][1] == B[k_][1]).all() for k_ in A) if set(A) == set(B) else False   # 全部の単位で正解の数が同じ（同じ判定になる版の印）
                for sn, ss in SEED_SETS.items():
                    As, Bs = sub(A, ss), sub(B, ss)
                    if sn == "0-2" or not As or (sn != "all" and len(As) == len(A)):
                        continue
                    effs.append(dict(a=a, b=b, c=c, K=K, seeds=sn, role_a=role_of(a), role_b=role_of(b), same_counts=bool(same), a_same_as=same_as[(c, K, a)],
                                     b_same_as=same_as[(c, K, b)], **eff(As, Bs, c, ("A", a, b, c, K, sn))))
    pd.DataFrame(effs).to_csv(sdir / "effects.csv", index=False)
    # ---- 仮定の表
    ck = csvs[csvs.version == "_check"]
    cols = [x for x in ck.columns if x.startswith(("intcorr_M_", "intperm_M_", "profcorr_M_", "levelcorr_M_", "topok_M_", "emocorr_M_", "ret_", "excess_", "level_M_"))
            or x in ("drift_M", "sdratio_M")]
    ass = ck.groupby([key, "K"])[cols].mean().reset_index().rename(columns={key: "c"})
    ass.merge(pd.DataFrame(ceil).groupby(["c", "K"]).mean().reset_index(), on=["c", "K"], how="left").to_csv(sdir / "assumptions.csv", index=False)
    pd.DataFrame([dict(T=T, c=c_, mod=m, ret=float(v[1][j]), **{f"ret_emo_{i}": float(v[0][j, i]) for i in range(4)}, **{f"ext_{i}": float(v[2][j, i]) for i in range(4)},
                       **{f"home_{i}": float(v[3][j, i]) for i in range(4)}) for (T, c_), v in sorted(ret_emo.items()) for j, m in enumerate(MODS)]).to_csv(sdir / "source_tables.csv", index=False)
    pd.DataFrame([dict(c=c, K=K, mod=m, emotion=i, w_prop=float(W[0, j, i]), w_orc=float(W[1, j, i]), w_K=float(W[2, j, i]), w_ext=float(W[3, j, i]), n_units=len(v))
                  for (c, K), v in sorted(wacc.items()) for W in [np.mean(v, 0)] for j, m in enumerate(MODS) for i in range(4)]).to_csv(sdir / "weights.csv", index=False)
    pd.DataFrame([dict(c=c, K=K, version=v, true=a, pred=b, n=float(M[a, b]), n_units=len(cells[(c, K, v)])) for (c, K, v), M in sorted(conf.items())
                  for a in range(4) for b in range(4)]).to_csv(                # n は全部の単位（分割 × 種）の合計
        sdir / "confusion.csv", index=False)
    es_ = csvs[csvs.version.astype(str).str.fullmatch(r"est_M_F6_mix\d+")].copy()
    es_["grid"] = es_.version.str.replace("est_M_F6_mix", "").astype(int)
    eg = es_.groupby([key, "K", "grid"])[["est_ll", "est_uar"]].mean().reset_index().rename(columns={key: "c"})
    eg["rank_ll"] = eg.groupby(["c", "K"]).est_ll.rank(ascending=False, method="min")
    eg["rank_uar"] = eg.groupby(["c", "K"]).est_uar.rank(ascending=False, method="min")
    eg["frozen"] = [bool((c, K) in pick and pick[(c, K)]["prop"] == i) for c, K, i in zip(eg.c, eg.K, eg.grid)]
    eg.to_csv(sdir / "est_grid.csv", index=False)
    wf = csvs[csvs.version.astype(str).str.fullmatch(r"wf\d+|tcal|temp|tempb|cf_D_l05")].copy()
    wf["eq_only_share"] = np.where(wf.wf_index.notna(), ((wf.wf_index == 0) & (wf.wf_n_tied == 1)).astype(float), np.nan)   # 等しい重みの表だけが最大だった単位（その単位では積と同じ判定）
    fc = [x for x in ("wf_index", "wf_n_tied", "wf_n_capped", "wf_uar_k", "wf_uar_k_eq", "eq_only_share", "w_A", "w_V", "w_T", "w_bound", "est_uar", "est_ll", "n_unused", "uar_raw", "uar_adj")
          if x in wf.columns]                                                # wf_uar_k は候補の中の最大の値で，上に偏る（テストの当たりの見積もりではない）
    wf.groupby([key, "K", "version"])[fc].mean().reset_index().rename(columns={key: "c"}).to_csv(sdir / "competitor_fit.csv", index=False)
    # ---- 実験 1・2 から写す値（B：実験 1 の集計の水準と区間．C：実験 2 の集計の水準と「実験 1 の提案 − 手法」の差）と，作り直した値との照合
    oth, dev_b = [], np.nan
    if tag == "eval":
        b = pd.read_csv(B_SRC, low_memory=False)
        b = b[(b.c == "ALL") & (b.student == "AVT") & b.K.isin(KS) & b.row.isin(list(B_ROWS))]
        for r in b.itertuples():
            mine = lv[(lv.c == r.T) & (lv.K == r.K) & (lv.version == B_ROWS[r.row]) & (lv.seeds == "all")]
            oth.append(dict(group="B", c=r.T, K=r.K, method=r.row, version=B_ROWS[r.row], uar=r.effect, lo=r.lo, hi=r.hi, se=r.se, n_units=r.n_units, seeds="0-9",
                            uar_here=(float(mine.uar.iloc[0]) if len(mine) else np.nan), source=str(B_SRC.relative_to(ROOT))))
        dev_b = float(np.max([abs(x["uar"] - x["uar_here"]) for x in oth])) if oth else np.nan      # 欠けた行があれば NaN になり，下で止まる
        if not allow_partial and len(oth) != len(CORP) * len(KS) * len(B_ROWS):
            raise SystemExit(f"実験 1 の集計から写す行が {len(oth)} 行（{len(CORP) * len(KS) * len(B_ROWS)} のはず）")
        if not allow_partial and not (dev_b < 1e-6):
            raise SystemExit(f"実験 1 の集計から写した水準と，この実験で作り直した水準が {dev_b:.2e} 違う")
        cc = pd.read_csv(C_SRC, low_memory=False)
        cc = cc[(cc.student == "AVT") & cc.K.isin(KS) & (cc["mode"] == "eval")]
        for r in cc.itertuples():
            oth.append(dict(group="C", c=r.T, K=r.K, method=r.method, version="", uar=r.uar_other, lo=np.nan, hi=np.nan, se=np.nan, n_units=r.n_units,
                            seeds=("0-2" if r.n_units == 15 else "0-9"), gradient=bool(r.gradient), appendix=bool(r.appendix), point=r.point,
                            descriptive_version=bool(r.descriptive_version), exp2_group=str(r.group),
                            exp1prop_minus_method=r.effect, exp1prop_minus_method_lo=r.lo, exp1prop_minus_method_hi=r.hi, uar_exp1prop_same_units=r.uar_prop,
                            source=str(C_SRC.relative_to(ROOT))))
    pd.DataFrame(oth).to_csv(sdir / "others.csv", index=False)
    c2 = csvs[csvs.version == "_check20"]
    json.dump(dict(n_units=len(files), teacher_dev_max=float(ck.teacher_dev.max()), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()),
                   adj_resid_max=float(csvs.adj_resid.max()), recheck_agree_min=float(c2.recheck_agree_min.min()), eq_agree_min=float(c2.eq_agree.min()),
                   psel_agree_min=float(c2.psel_agree.min()),
                   others_B_dev_max=dev_b, others_sha256=({str(q.relative_to(ROOT)): dai15.sha(q) for q in (B_SRC, C_SRC)} if tag == "eval" else {}),
                   unit_commits=ucom, summary_commit=RUN["commit"],
                   sec_total=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())), open(sdir / "convergence.json", "w"), indent=1)
    print(f"summary：単位 {len(files)}，差の行 {len(effs)}，水準の行 {len(lrows)}．作り直した提案の判定の一致の最小 {c2.recheck_agree_min.min():.6f}，"
          f"等しい重みの表と積の一致の最小 {c2.eq_agree.min():.6f}，指数の積の作り直しの一致の最小 {c2.psel_agree.min():.6f}，"
          f"切片の反復が上限に当たった（組, 表）の数 {int(csvs[csvs.version == f'wf{N_WF}'].wf_n_capped.fillna(0).sum())}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    assert len(GRID) == 16 and GRID[2] == (0.0, 1.0, "eq") and GRID[14] == (float("inf"), 1.0, "eq")
    for i, (n0, b, p) in enumerate(GRID):
        assert GRID[same_but(i, beta=0.0)] == (n0, 0.0, p) and GRID[same_but(i, n0=0.0)] == (0.0, b, p)
    assert WS.shape == (N_WF + 1, 3, 4) and np.allclose(WS.sum(1), 3.0) and np.allclose(WS[0], 1.0) and WS.min() >= 0
    assert np.allclose(WS[:11], np.concatenate([np.ones((1, 3, 4)), 3.0 * np.random.default_rng([SEED, 1]).dirichlet(np.ones(3), size=(N_WF, 4)).transpose(0, 2, 1)])[:11])
    rng = np.random.default_rng(0)
    # 束の切片のずらしが，1 つずつのずらし（実験 16 の関数）と同じ値になる
    n = 300
    P = rng.dirichlet(np.ones(4) * 0.7, size=(3, n))
    LM = np.stack([P16.centered(p) for p in P])
    tgt = np.exp(P16.norm_log(LM.sum(0))).mean(0)
    Sp = np.einsum("rky,kny->rny", WS[:6], LM)
    Sp = Sp - scipy.special.logsumexp(Sp, axis=-1, keepdims=True)
    db, ncap = match_shift_batch(torch.tensor(Sp), torch.tensor(tgt))
    db = db.numpy()
    assert ncap == 0
    for r_ in range(6):
        d1, rs = P16.match_shift(Sp[r_], tgt)
        assert np.abs(db[r_] - d1).max() < 1e-9 and rs < 1e-6, (r_, np.abs(db[r_] - d1).max())
    assert np.abs(db[0]).max() == 0.0                                       # 等しい重みの表は切片が動かない（積と同じ点）
    # 重みの表を探す：K 本の UAR の式が 1 つずつの計算と合う．等しい重みの表の値は，積を同じ道に通した値と同じ
    K = 5
    yk = np.repeat(np.arange(4), K)
    fold = dai15.kfold_of(yk)
    Ph = rng.dirichlet(np.ones(4), size=(3, 4 * K))
    Ph[:, np.arange(4 * K), yk] += 0.4
    Ph /= Ph.sum(-1, keepdims=True)
    LMh = np.stack([P16.centered(p) for p in Ph])
    LMpj = [np.stack([P16.centered(p) for p in rng.dirichlet(np.ones(4) * 0.7, size=(3, n))]) for _ in range(NF)]
    ua, ncap = wf_search(LMh, yk, fold, LMpj)
    assert ua.shape == (N_WF + 1,) and ncap == 0
    for r_ in (0, 1, 7, N_WF):
        pred = np.empty(4 * K, int)
        for j in range(NF):
            te = np.flatnonzero(fold == j)
            dl, _ = P16.match_shift(P16.norm_log(np.einsum("ky,kny->ny", WS[r_], LMpj[j])), np.exp(P16.norm_log(LMpj[j].sum(0))).mean(0))
            pred[te] = (P16.norm_log(np.einsum("ky,kny->ny", WS[r_], LMh[:, te])) + dl).argmax(1)
        assert abs(ua[r_] - 100 * S.uar(yk, pred)) < 1e-9, r_
    assert abs(ua[0] - 100 * S.uar(yk, LMh.sum(0).argmax(1))) < 1e-9          # 表 0 は積の判定（切片は動かない）
    W, tied = wf_pick(ua, 10)
    assert np.allclose(W.sum(0), 3.0) and (tied <= 10).all() and np.allclose(W, WS[tied].mean(0)) and ua[tied[0]] >= ua[:11].max() - 1e-9
    W0, t0 = wf_pick(np.r_[50.0, np.zeros(N_WF)], 1000)
    assert list(t0) == [0] and np.allclose(W0, 1.0)
    W1, t1 = wf_pick(np.r_[50.0, 50.0, 60.0, 60.0, np.zeros(N_WF - 3)], 1000)
    assert list(t1) == [2, 3] and list(wf_pick(np.r_[50.0, 50.0, 60.0, 60.0, np.zeros(N_WF - 3)], 1)[1]) == [0, 1]
    # 重みが全部 1 なら積と同じ判定．K 本で合わせた重みは範囲の中で，分類器が 1 つだけ当たるならその重みが最大
    assert (np.einsum("ky,kny->ny", np.ones((3, 4)), LM).argmax(1) == np.log(P).sum(0).argmax(1)).all()
    Pg = rng.dirichlet(np.ones(4), size=(3, 40))
    yg = np.repeat(np.arange(4), 10)
    Pg[0, np.arange(40), yg] += 1.0
    Pg /= Pg.sum(-1, keepdims=True)
    Lg = np.stack([P16.centered(p) for p in Pg])
    for fb in TEMP_MODES.values():
        w = temp_w(Lg, yg, fb)
        assert w.shape == (3,) and (w >= 0).all() and (w <= P16.W_MAX).all() and w.argmax() == 0, w
    assert np.allclose(temp_w(Lg[:1], yg, "sep"), temp_w(Lg[:1], yg, "joint"), atol=1e-5)          # 分類器が 1 つなら 2 つの合わせ方は同じ
    # 相性を home から取る版：home が ext の定数倍（残り方が 1）なら提案の表と同じ
    DK, De = rng.uniform(0.2, 2.0, size=(3, 4)), rng.uniform(0.2, 2.0, size=(3, 4))
    for n0, beta, prior in GRID:
        if beta == 1.0:
            lev = P16.mix_rel(DK, De, 20, n0, 0.0, prior)
            assert np.allclose(np.maximum(0.0, lev + (De - De.mean(1, keepdims=True))), P16.mix_rel(DK, De, 20, n0, 1.0, prior))
    assert role_of("prop") == "proposal" and role_of("wf10@raw") == "A" and role_of("b_kd") == "B" and role_of("M_F6_orc") == "oracle"
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "freeze", "units", "summary"))
    ap.add_argument("--tag", default="eval", choices=("eval", "pseudo"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if args.stage == "freeze":
        stage_freeze()
        return
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある")
    if prod and args.tag != "eval":
        sys.exit("本番の置き場は評価の回だけ")
    if not prod and args.tag == "eval":
        sys.exit("評価の回は本番の置き場にだけ書く（検査は --tag pseudo で）")
    if prod and not (dai1.committed_clean(DESIGN) and dai1.committed_clean(FROZEN)):
        sys.exit("設計の文書か固定のファイルがコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.allow_partial):
        sys.exit("本番の置き場では分割と種を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    P16.KS = KS                                                             # 実験 16 の単位の計算を K = 5・10・25 で回す
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, tag=args.tag, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, python=platform.python_version(), this_script_sha256=dai15.sha(Path(__file__).resolve()),
                scripts={n: dai15.sha(HERE / n) for n in SCRIPTS if (HERE / n).exists()}, design_sha256=(dai15.sha(DESIGN) if DESIGN.exists() else ""),
                frozen_sha256=(dai15.sha(FROZEN) if FROZEN.exists() else ""), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1],
                inputs_sha256={**{f"dai1_source/{q.name}": dai15.sha(q) for q in sorted((OUT1 / "source").glob("*.npz"))},
                               **{f"splits/{q.name}": dai15.sha(q) for q in sorted((ROOT / "splits").glob("*.csv"))}},
                KS=KS, GRID=[list(map(str, g_)) for g_ in GRID], rng_seed=SEED, settings=SETTINGS, wf_tables_sha256=hashlib.sha256(WS.tobytes()).hexdigest(),
                consts=dict(D_MIN=P16.D_MIN, EPS=P16.EPS, AUC_CLIP=P16.AUC_CLIP, TAU=P16.TAU, TAU_F8=P16.TAU_F8, W_MAX=P16.W_MAX, ADJ_ITERS=P16.ADJ_ITERS, ADJ_TOL=P16.ADJ_TOL,
                            CHECK_TEACHER=P16.CHECK_TEACHER, GRAD_TOL=P16.GRAD_TOL, NF=NF))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        stage_units(out, args.tag, args.targets, args.folds, args.seeds)
    else:
        stage_summary(out, args.tag, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
