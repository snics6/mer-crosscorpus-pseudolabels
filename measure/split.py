"""話者分割の正本（data/README.md 取り決め 5）。**評価する全コードはここを通す。**

学習側と評価側に同じ人物が現れてはならない。ところが `speaker_id` が
そのまま人物を指しているとは限らないので，**コーパスごとに 2 つの補正を入れる。**
補正の根拠は `measure/check_speakers.py` の診断（`results/speaker_audit.csv`）。

## 補正 1：AFEW は俳優名を使う（作品名の補正は不要になった）

前身のマニフェストは `speaker_id` に**作品名**を入れていた。作品名で分けると
**41 人の俳優が複数の作品に出ており，のべ 373 本（AFEW の 32%）が漏れる**
（Daniel Radcliffe が 10 作品，Rupert Grint 6，Julia Roberts 6 など）。

**`features/build_manifest_afew.py` が公式注釈の `NameOfActor` を使うようにしたので，
この補正は要らなくなった。**話者は俳優そのものである（俳優 345 人．名前の付かない 4 本は作品名）。

**残る限界：4 本だけ XML と対応がつかず，作品名に落としている。**

## 補正 2：MELD の合同発話を落とす

`Ross_and_Rachel` のような id が 8 種ある。**`Ross` を評価側に取り置いても
`Ross_and_Rachel` が学習側に残れば Ross の声と顔が漏れる。**該当は 8 発話
（MELD の 0.07%）なので落とす。

**`A_Waiter` と `The_Waiter` のような一般名詞の役名の衝突（118 発話）は落とさない。**
同じ役名の別の端役であり，同一人物とは限らない。落とすと無駄に捨てることになる。

## 使い方

    from split import speaker_groups, speaker_folds

    g = speaker_groups(df, "MELD")          # 話者群．NaN は除外する発話
    f = speaker_folds(g, k=5, seed=0)       # 群を壊さずに k 個へ配る
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
import common  # noqa: E402

# 作品名を話者に使っているコーパス。**いまは無い**（AFEW は俳優名になった）。
# 将来そういうコーパスを足したときのために仕組みは残す
FILM_AS_SPEAKER: tuple[str, ...] = ()
STEM_MIN_LEN = 6


def _film_stem(name: str) -> str:
    """作品名から系列の幹を取る。区切りが無い綴りにも効くよう，
    先頭の大文字ではじまる語をつなげずに最初の 1 語だけを見る。"""
    head = re.split(r"[_\-]", name)[0]
    # 'HarryPotterDeathlyHallows' のように区切りが無い綴りも拾う
    m = re.match(r"[A-Z][a-z]+[A-Z][a-z]+", head)
    if m and len(head) > STEM_MIN_LEN:
        head = m.group(0)
    return head.lower() if len(head) >= STEM_MIN_LEN else name.lower()


def _tiebreak(group: str, seed: int) -> int:
    """同数の群を並べるときの決定的な順序。**群の名前と seed だけで決まる。**
    seed を変えれば順序は変わるが，入力の行順には一切依存しない。"""
    h = hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def speaker_groups(df: pd.DataFrame, corpus: str) -> pd.Series:
    """発話ごとの話者群を返す。**NaN の行は評価から外す**（漏れるため）。"""
    sp = df["speaker_id"].astype(str)

    if corpus in FILM_AS_SPEAKER:
        return sp.map(_film_stem)

    if corpus == "MELD":
        joint = sp.str.contains("_and_", regex=False)
        return sp.where(~joint, other=pd.NA)

    return sp


def speaker_folds(groups: pd.Series, k: int = 5, seed: int = 0) -> pd.Series:
    """話者群を壊さずに k 個の袋へ配る。件数が揃うよう，大きい群から
    そのとき最も軽い袋へ入れる（貪欲法）。同数のときは seed で決める。

    返すのは発話ごとの袋の番号（0〜k−1）。**群が NaN の行は −1**。
    """
    valid = groups.notna()
    vc = groups[valid].value_counts()
    # **同数の群の順序は群の名前から決める。入力の行順に依存させない。**
    # 前は `rng.random()` を key にしていたが，sorted は key を入力順に評価する
    # ので，**乱数がどの群に当たるかが入力順で変わっていた**。実測では MELD の
    # 283 群中 189 群が，行を並べ替えるだけで違う袋に移った（件数は同じ）。
    # モダリティごとに特徴の行順が違えば，**同じ seed でも別々の分割**になり，
    # モダリティ間で結果を比べられなくなる。
    order = sorted(vc.index, key=lambda g: (-int(vc[g]), _tiebreak(g, seed)))

    load = np.zeros(k)
    assign: dict[str, int] = {}
    for g in order:
        j = int(load.argmin())
        assign[g] = j
        load[j] += int(vc[g])

    out = groups.map(assign)
    return out.fillna(-1).astype(int)


def check_disjoint(groups: pd.Series, folds: pd.Series) -> None:
    """1 つの話者群が 2 つ以上の袋にまたがっていないことを確かめる。
    またがっていれば，その時点で落とす。**黙って進めない。**"""
    ok = groups.notna() & (folds >= 0)
    bad = [g for g, s in pd.DataFrame({"g": groups[ok], "f": folds[ok]}).groupby("g")["f"]
           if s.nunique() > 1]
    if bad:
        raise SystemExit(f"話者群が袋をまたいでいる: {bad[:5]}")


def main() -> None:
    """全コーパスで分割を作り，漏れが無いことを確かめて表にする。"""
    rows = []
    for c in common.CORPORA_ALL:
        df = common.load_manifest(c)
        g = speaker_groups(df, c)
        f = speaker_folds(g, k=5, seed=0)
        check_disjoint(g, f)

        dropped = int((f < 0).sum())
        sizes = [int((f == j).sum()) for j in range(5)]
        labels_ok = all(df.loc[f == j, "label_4"].nunique() == 4 for j in range(5))
        rows.append({
            "corpus": c,
            "発話": len(df),
            "元の話者": df["speaker_id"].nunique(),
            "話者群": int(g.nunique()),
            "除外": dropped,
            "5分割": f"{min(sizes)}〜{max(sizes)}",
            "4ラベル揃う": "○" if labels_ok else "×",
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    dest = common.ROOT / "results" / "split_summary.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\n書いた: {dest}")
    print("話者群が袋をまたいでいないことを全コーパスで確認した。")


if __name__ == "__main__":
    main()
