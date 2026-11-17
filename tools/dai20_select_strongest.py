"""提案と「K 本で最も強い分類器を 1 つ選ぶ」相手の比べ（計算の後の解析．2026-11-16 夜．学び直しはしない）．第 2 版．

動機（研究者との相談，2026-11-16 夜）：論文の問い 2 の比べる相手（実験 20 の A の組）には，「K 本のラベルで最も強い分類器を
1 つ選び，その分類器の答えだけを使う」相手が無い．AFEW では顔の分類器 1 つの UAR が積より高い K があったので（論文の下書き 7.4 節），
査読者は「最強の 1 つを選べば済むのでは」と問いうる．この台本は，保存した出力だけでこの相手の UAR を作り，提案（推定時の答え）と比べる．
第 2 版は台本の検査 checks/dai20_select_strongest/code.md（条件つき OK）を受けて直した（第 2 版の再検査 code_v2.md を受けて，ret の raw の判定の照合で float32 の丸めによる同点を一致と数え，感情ごとの出力に件数の列を足した）：切片のずらしを両者にそろえた層を主にし，
選び方を 2 つ足し，感情ごと・種の部分集合・単位ごとの選びを書き出し，出力の安全の仕組みを足した．

範囲：積を作る声・顔・言葉の分類器（融合の分類器の疑似ラベルで学び直した ret_A，ret_V，ret_T）からの選びに限る．K 本だけで学んだ
基準の分類器からの選びと CREMA-D は範囲の外（保存の名前と固定の点と g_eff の作りが違うため）．

選ぶ相手（どれも ret_A，ret_V，ret_T の 1 つの答えを使う）：
  sel_K    ：K 本のラベルで測った強さ（D_M_K_K<K> の 4 感情の平均．提案が K 本で測る水準の材料．tools/dai20_strongest_from_k.py の
              top_k と同じ）が最大の分類器．新しいコーパスで手元にある情報だけで選ぶ．
  sel_Kuar ：K 本の上の交差の答え（pcf_<m>_K<K>．組を抜いて作り直した分類器が K 本に付けた確率）の UAR が最大の分類器．同点は，
              同点の中で D_M_K の 4 感情の平均が最大のもの，それでも同点なら A，V，T の順．手元にある情報だけで選ぶ．
  sel_ext  ：ソースのコーパスだけで測った強さ（SR_M_ext の 4 感情の平均）が最大の分類器．ソースだけで決まるので，コーパスごとに
              単位と K によらず同じ分類器になる（結果の文では「そのコーパスで決まった 1 つの分類器」と書く）．
  sel_orc  ：K 本を除いた候補の全ラベルで測った強さ（D_M_orc_K<K>）が最大の分類器．新しいコーパスでは手元に無い情報で選ぶ参考の版．
              候補の発話の d′ で選ぶので，テストで選んだ最良の 1 つの上限ではない．
  sel_test ：単位ごとにテストのラベルで UAR が最大の分類器（参考の上限．同じ層の判定の中で選ぶ．同点は A，V，T の順）．
  sel_K，sel_ext，sel_orc の同点は A，V，T の順（np.argmax）．同点の数は出力に書く．

2 つの層（切片のずらしをそろえる．提案の推定時の答えは，ラベルを使わずに候補の上の確率の平均を積に合わせて切片をずらした答え）：
  adj 層（主）：単独の分類器にも exp/dai20_eval.py の add_extra の put と同じずらし（P16.match_shift，目標は積の候補の上の確率の
                平均）を当てた判定 sel_*@adj と，提案 prop（pred_adj_K<K>_M_F6_mix<i>）．積 prod はずらしの量が 0 なので raw と同じ．
  raw 層      ：どちらもずらさない．提案 prop@raw（pred_raw_K<K>_M_F6_mix<i>）と単独の分類器の raw の判定 sel_*．
主に読む対（値を見る前に決めた）：adj 層の prop − sel_K@adj．ほかの対は参考に並べる．

比べの一覧：adj 層 prop − sel_{K,Kuar,ext,orc,test}@adj，prod − sel_K@adj．raw 層 prop@raw − sel_{K,Kuar,ext,orc,test}，prod − sel_K．
区間は実験 20 と同じ作り（stage_summary の eff と同じ ST.effect_stats，同じ g_eff，鍵 ("A", a, b, c, K, 種の組) と ("L", c, K, v, 種の組)，
("E", c, K, v, 感情)）．種の組は集計と同じ規則で，差は all・0-1・2-9，水準は all・0-2．感情ごとの再現率は all で全部の版について出す．

自己検査（どれかが合わなければ止まり，何も書かない）：
  (1) prop − prod と prod・prop・prop@raw の水準を作り直し，results/dai20_eval/summary/effects.csv・levels.csv と 1e-9 以内で一致．
  (2) sel_K・sel_orc・sel_ext の分類器が results/dai20_eval/posthoc/strongest_from_k.csv の top_k・top_orc・top_ext と全行で一致．
  (3) adj 層のずらしの式で avg の adj の判定を作り直し，保存の pred_adj_K<K>_avg と全単位で一致．
記述の解析で予言は置かない．同じコーパスの 3 つの K は同じテストと入れ子の K 本を共有するので独立でない．区間は多重比較の補正をしない．

出力（上書きしない．--targets で一部のコーパスだけを流すときは既定の置き場に書かない）：results/dai20_eval/posthoc/ の
select_strongest_effects.csv（差），select_strongest_levels.csv（水準と選んだ分類器の数），select_strongest_per_emotion.csv（感情ごとの
再現率），select_strongest_picks.csv（単位 × K ごとに選んだ分類器），select_strongest_provenance.json．
使い方：.venv/bin/python tools/dai20_select_strongest.py [--out-dir 一時の置き場 --targets AFEW,MELD]
  台本をコミットしてから本番を流す（dirty が記録に残る）．
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai20_eval as E  # noqa: E402

ST, P12, P16, S = E.ST, E.P12, E.P16, E.S
UNITS = ROOT / "results" / "dai20_eval" / "units"
SUMM = ROOT / "results" / "dai20_eval" / "summary"
POST = ROOT / "results" / "dai20_eval" / "posthoc"
MODS = ("A", "V", "T")
KS = E.KS
SELS = ("sel_K", "sel_Kuar", "sel_ext", "sel_orc", "sel_test")
CONTRASTS = ([("prop", f"{s}@adj") for s in SELS] + [("prod", "sel_K@adj")]
             + [("prop@raw", s) for s in SELS] + [("prod", "sel_K")])
MAIN = ("prop", "sel_K@adj")
NAMES = ("select_strongest_effects.csv", "select_strongest_levels.csv", "select_strongest_per_emotion.csv",
         "select_strongest_picks.csv", "select_strongest_provenance.json")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def argmax_first(v: np.ndarray) -> tuple[int, bool]:
    i = int(np.argmax(v))
    return i, bool(np.sum(np.isclose(v, v[i], rtol=0, atol=1e-12)) > 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(POST))
    ap.add_argument("--targets", default=",".join(E.CORP))
    args = ap.parse_args()
    out = Path(args.out_dir)
    targets = tuple(args.targets.split(","))
    if not set(targets) <= set(E.CORP) or len(set(targets)) != len(targets):
        raise SystemExit(f"コーパスの指定がおかしい：{targets}")
    if set(targets) != set(E.CORP) and out.resolve() == POST.resolve():
        raise SystemExit("一部のコーパスだけを流すときは --out-dir に一時の置き場を指定する（既定の置き場に一部だけの出力を書かない）")
    out.mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        if (out / n).exists():
            raise SystemExit(f"{out / n} が既にある（上書きしない）")
    frozen = E.load_frozen()
    sfk = pd.read_csv(POST / "strongest_from_k.csv")
    ucommits = set()
    cells, picks, used = {}, [], []
    dev_avg = 0
    n_argmax_tie = 0
    for T in targets:
        files = sorted(UNITS.glob(f"{T}_ALL_f*_r*.npz"))
        if len(files) != 50:
            raise SystemExit(f"{T}：単位が {len(files)} 個（見込み 50）")
        for p in files:
            f, r = (int(x[1:]) for x in p.stem.split("_")[2:4])
            ucommits |= set(pd.read_csv(p.with_suffix(".csv"), usecols=["commit"], low_memory=False).commit.astype(str))
            z = np.load(p, allow_pickle=False)
            used.append(p)
            yt, cl, yp = z["y_test"].astype(int), z["cl_test"], z["y_pool"].astype(int)
            for K in KS:
                i_fz = frozen[T][K][0]
                kpos = z[f"kpos_K{K}"]
                yk = yp[kpos]
                Pt = [z[f"p_test_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
                Pp = [z[f"p_pool_ret_{m}_K{K}"].astype(np.float64) for m in MODS]
                Ph = [z[f"pcf_{m}_K{K}"].astype(np.float64) for m in MODS]
                LMp = np.stack([P16.centered(q) for q in Pp])
                target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)          # 積の，候補の上の確率の平均（add_extra と同じ）

                def adj(St: np.ndarray, Sp: np.ndarray) -> np.ndarray:
                    dl, _ = P16.match_shift(Sp, target)
                    return (St + dl).argmax(1)

                # 自己検査 (3)：avg の adj の判定を同じ式で作り直す
                rec = adj(P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0)))
                dev_avg = max(dev_avg, int((rec != z[f"pred_adj_K{K}_avg"].astype(int)).sum()))
                raw = {m: z[f"pred_raw_K{K}_ret_{m}"].astype(int) for m in MODS}
                for j, m in enumerate(MODS):                                   # float32 に丸めた確率で最大と並ぶだけの発話は一致として数え，数を記録する
                    hit = raw[m] == Pt[j].argmax(1)
                    tie_ok = Pt[j][np.arange(len(raw[m])), raw[m]] == Pt[j].max(1)
                    if not (hit | tie_ok).all():
                        raise SystemExit(f"{p.name} K{K}：ret_{m} の raw の判定が確率の argmax と違う")
                    n_argmax_tie += int((~hit & tie_ok).sum())
                shifted = {m: adj(P16.lg(Pt[j]), P16.lg(Pp[j])) for j, m in enumerate(MODS)}
                dk = z[f"D_M_K_K{K}"].astype(float).mean(1)
                strength = {"sel_K": dk, "sel_ext": z["SR_M_ext"].astype(float).mean(1), "sel_orc": z[f"D_M_orc_K{K}"].astype(float).mean(1)}
                row = dict(T=T, fold=f, seed=r, K=K)
                choice = {}
                for v, s in strength.items():
                    if s.shape != (3,) or not np.isfinite(s).all():
                        raise SystemExit(f"{p.name} K{K} {v}：強さの形か値がおかしい")
                    i, tie = argmax_first(s)
                    choice[v] = i
                    row[f"{v}_mod"], row[f"{v}_tie"] = MODS[i], tie
                uk = np.array([100 * S.uar(yk, Ph[j].argmax(1)) for j in range(3)])
                cand = np.flatnonzero(np.isclose(uk, uk.max(), rtol=0, atol=1e-9))
                i = int(cand[np.argmax(dk[cand])])                             # 同点の中で D_M_K の平均が最大，それでも同点なら A，V，T の順
                choice["sel_Kuar"] = i
                row["sel_Kuar_mod"], row["sel_Kuar_tie"] = MODS[i], bool(len(cand) > 1)
                row.update({f"kuar_{m}": float(uk[j]) for j, m in enumerate(MODS)})
                ref = sfk[(sfk["T"] == T) & (sfk.fold == f) & (sfk.seed == r) & (sfk.K == K)]
                if (len(ref) != 1 or ref.top_k.iloc[0] != row["sel_K_mod"] or ref.top_orc.iloc[0] != row["sel_orc_mod"]
                        or ref.top_ext.iloc[0] != row["sel_ext_mod"]):
                    raise SystemExit(f"{p.name} K{K}：選んだ分類器が strongest_from_k.csv と違う")
                preds = {"prop": z[f"pred_adj_K{K}_M_F6_mix{i_fz}"].astype(int), "prop@raw": z[f"pred_raw_K{K}_M_F6_mix{i_fz}"].astype(int),
                         "prod": z[f"pred_raw_K{K}_prod"].astype(int)}
                for layer, judg, suf in (("raw", raw, ""), ("adj", shifted, "@adj")):
                    for v in ("sel_K", "sel_Kuar", "sel_ext", "sel_orc"):
                        preds[f"{v}{suf}"] = judg[MODS[choice[v]]]
                    ut = np.array([100 * S.uar(yt, judg[m]) for m in MODS])
                    it = int(np.argmax(ut))
                    preds[f"sel_test{suf}"] = judg[MODS[it]]
                    row[f"sel_test_{layer}_mod"] = MODS[it]
                    row[f"sel_test_{layer}_tie"] = bool(np.sum(np.isclose(ut, ut[it], rtol=0, atol=1e-9)) > 1)
                picks.append(row)
                for v, pr in preds.items():
                    if pr.shape != yt.shape:
                        raise SystemExit(f"{p.name} K{K} {v}：判定の長さが違う")
                    P12._add(cells.setdefault((T, K, v), {}), (f, r), ST.counts(yt, pr, cl), f"{p.stem} {v}")
    if len(ucommits) != 1:
        raise SystemExit(f"単位の csv の commit が 1 通りでない：{ucommits}")
    units_commit = next(iter(ucommits))[:7]
    if dev_avg:
        raise SystemExit(f"自己検査 (3)：avg の adj の判定を作り直せなかった（違う発話 {dev_avg}）．何も書かない")
    P12._finish(cells)
    geff = {c: ST.g_eff_of(c)[0] for c in targets}

    def eff(A, B, c, k_):                                                   # exp/dai20_eval.py の stage_summary の eff と同じ
        if set(A) != set(B):
            raise SystemExit(f"差の単位がそろわない：{k_}")
        es = ST.effect_stats(A, B, geff[c], E.seed_rng(*k_))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    n_folds=es["n_folds"], folds_dropped=es.get("folds_dropped", ""), v_cluster=es.get("v_jk", np.nan), v_seed=es.get("v_seed", np.nan),
                    v_seed_all=es.get("v_seed_all", np.nan),
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))

    def zero(A):
        return {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}

    def only(A, i):
        m = np.zeros(4)
        m[i] = 1.0
        return {k_: (x[0], x[1] * m, x[2]) for k_, x in A.items()}

    def sub(A, seeds):
        return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}

    # ---- 自己検査 (1)
    effs0, lev0 = pd.read_csv(SUMM / "effects.csv"), pd.read_csv(SUMM / "levels.csv")
    dev = 0.0
    for c in targets:
        for K in KS:
            e = eff(cells[(c, K, "prop")], cells[(c, K, "prod")], c, ("A", "prop", "prod", c, K, "all"))
            ref = effs0[(effs0.a == "prop") & (effs0.b == "prod") & (effs0.c == c) & (effs0.K == K) & (effs0.seeds == "all")]
            if len(ref) != 1:
                raise SystemExit(f"effects.csv に prop − prod の行が無い：{c} K{K}")
            dev = max(dev, *(abs(e[k] - float(ref[k].iloc[0])) for k in ("est", "lo", "hi")))
            for v in ("prop", "prod", "prop@raw"):
                l_ = eff(cells[(c, K, v)], zero(cells[(c, K, v)]), c, ("L", c, K, v, "all"))
                rl = lev0[(lev0.c == c) & (lev0.K == K) & (lev0.version == v) & (lev0.seeds == "all")]
                if len(rl) != 1:
                    raise SystemExit(f"levels.csv に {v} の行が無い：{c} K{K}")
                dev = max(dev, *(abs(l_[k] - float(rl[{"est": "uar"}.get(k, k)].iloc[0])) for k in ("est", "lo", "hi")))
    if dev > 1e-9:
        raise SystemExit(f"自己検査 (1)：実験 20 の集計を作り直せなかった（ずれの最大 {dev:.2e}）．何も書かない")
    print(f"自己検査：(1) 作り直しのずれの最大 {dev:.1e}．(2) 選んだ分類器は strongest_from_k.csv と全 {len(picks)} 行で一致．"
          f"(3) avg の adj の判定の作り直しは全単位で一致", flush=True)
    # ---- 本体
    erows, lrows, emrows = [], [], []
    versions = ("prop", "prop@raw", "prod") + SELS + tuple(f"{s}@adj" for s in SELS)
    for c in targets:
        for K in KS:
            for a, b in CONTRASTS:
                A, B = cells[(c, K, a)], cells[(c, K, b)]
                same = all((A[k_][1] == B[k_][1]).all() for k_ in A)
                for sn, ss in E.SEED_SETS.items():
                    As, Bs = sub(A, ss), sub(B, ss)
                    if sn == "0-2" or not As or (sn != "all" and len(As) == len(A)):
                        continue
                    erows.append(dict(a=a, b=b, c=c, K=K, seeds=sn, main=bool((a, b) == MAIN), same_counts=bool(same),
                                      **eff(As, Bs, c, ("A", a, b, c, K, sn))))
            pk = pd.DataFrame([x for x in picks if x["T"] == c and x["K"] == K])
            for v in versions:
                dd = cells[(c, K, v)]
                base = v.split("@")[0]
                extra = {}
                if base in SELS:
                    col = f"{base}_mod" if base != "sel_test" else f"sel_test_{'adj' if v.endswith('@adj') else 'raw'}_mod"
                    tcol = col.replace("_mod", "_tie")
                    extra = {f"n_{m}": int((pk[col] == m).sum()) for m in MODS} | {"n_tie": int(pk[tcol].sum())}
                for sn, ss in E.SEED_SETS.items():
                    ds = sub(dd, ss)
                    if not ds or (sn != "all" and len(ds) == len(dd)) or sn not in ("all", "0-2"):
                        continue
                    l_ = eff(ds, zero(ds), c, ("L", c, K, v, sn))
                    lrows.append(dict(c=c, K=K, version=v, seeds=sn, uar=l_["est"], lo=l_["lo"], hi=l_["hi"], se=l_["se"], df=l_["df"],
                                      method=l_["method"], n_fr=l_["n_fr"], **extra))
                full = all((x[2].sum(0) > 0).all() for x in dd.values())
                for i in range(4):
                    r_ = eff(only(dd, i), zero(dd), c, ("E", c, K, v, i)) if full else None
                    emrows.append(dict(c=c, K=K, version=v, emotion=i, recall=(4 * r_["est"] if full else np.nan),
                                       lo=(4 * r_["lo"] if full else np.nan), hi=(4 * r_["hi"] if full else np.nan),
                                       n_fr=(r_["n_fr"] if full else 0), n_units=len(dd), n=float(sum(x[2][:, i].sum() for x in dd.values()))))
    cm, dt = E.dai1.git_commit()
    tagd = dict(commit=cm, dirty=dt, units_commit=units_commit)
    pd.DataFrame(erows).assign(**tagd).to_csv(out / NAMES[0], index=False)
    pd.DataFrame(lrows).assign(**tagd).to_csv(out / NAMES[1], index=False)
    pd.DataFrame(emrows).assign(**tagd).to_csv(out / NAMES[2], index=False)
    pd.DataFrame(picks).assign(**tagd).to_csv(out / NAMES[3], index=False)
    prov = dict(script="tools/dai20_select_strongest.py", script_sha256=sha16(Path(__file__)), commit=cm, dirty=dt, targets=list(targets),
                units=len(used), units_commit=units_commit,
                units_sha256_of_names=hashlib.sha256("".join(sorted(p.name + sha16(p) for p in used)).encode()).hexdigest()[:16],
                inputs={str(x.relative_to(ROOT)): sha16(x) for x in (SUMM / "effects.csv", SUMM / "levels.csv", POST / "strongest_from_k.csv",
                                                                       ROOT / "exp" / "dai20_eval.py")},
                selfcheck_max_dev=dev, selfcheck_avg_adj_mismatch=dev_avg, ret_raw_argmax_float32_ties=n_argmax_tie, main_contrast=list(MAIN))
    (out / NAMES[4]).write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    print(f"書いた：{out}（{', '.join(NAMES)}）", flush=True)


if __name__ == "__main__":
    main()
