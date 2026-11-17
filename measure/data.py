"""測定で使う特徴の読み込み。**すべての測定コードがここを通る。**

## 群（group）という単位

コーパスをそのまま使わず，**群**を単位にする。IEMOCAP は即興と台本を分ける
（`data/README.md` 取り決め 6：台本では 10 人全員が同じ台詞を読むので，言葉の分類器が
台詞を覚えてしまう）。

    IEMOCAP-impro / IEMOCAP-script / MOSEI / MELD / AFEW / RAVDESS / CREMA-D

## ラベルはマニフェストから貼り直す

特徴の npz には**抽出した時点のラベルが焼き付いている。**マニフェストを直しても
古いままなので，必ず貼り直す。前身のマニフェストは IEMOCAP の 57% に誤ったラベルを
付けていた（`data/README.md` 3 節「前身のマニフェストにあった誤り」）。**ラベルの正本はマニフェスト 1 つに限る。**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from split import speaker_groups  # noqa: E402

# 群 → (コーパス, 対話の種類の絞り込み)
GROUP_SPEC: dict[str, tuple[str, str | None]] = {
    "IEMOCAP-impro": ("IEMOCAP", "impro"),
    "IEMOCAP-script": ("IEMOCAP", "script"),
    "MOSEI": ("MOSEI", None),
    "MELD": ("MELD", None),
    "AFEW": ("AFEW", None),
    "RAVDESS": ("RAVDESS", None),
    "CREMA-D": ("CREMA-D", None),
}
GROUPS_ALL = tuple(GROUP_SPEC)

# モダリティ → (第 1 エンコーダ, 第 2 エンコーダ)
MODALITIES: dict[str, tuple[str, str]] = {
    "video": ("video", "video_clip"),
    "audio": ("audio_wavlm", "audio_hubert"),
    "text": ("text_roberta", "text_deberta"),
}

# 特徴が入っている配列の名前。映像だけ平均と標準偏差に分かれている
FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "video": ("X_mean", "X_std"), "video_clip": ("X_mean", "X_std"),
    "audio_wavlm": ("X",), "audio_hubert": ("X",),
    "text_roberta": ("X",), "text_deberta": ("X",),
}

# 言葉が使えるのは書き起こしのある 4 コーパスの 5 群（AFEW は Whisper の書き起こし．
# 2026-11-04 に足した）。RAVDESS・CREMA-D は読む文が 2 種類・12 種類しかないので使わない
GROUPS_WITH_TEXT = ("IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW")


def has_modality(group: str, modality: str) -> bool:
    if modality == "text":
        return group in GROUPS_WITH_TEXT
    if modality == "video" and group == "RAVDESS":
        return False        # 発話一覧に映像の場所（path_video）を入れていない．生データは取得済みで，入れるかは docs/researchvision.md 7 節
    return True


def load_group(group: str, kind: str, with_std: bool = False
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """1 群ぶんの (表, 特徴行列) を返す。**成功した行だけ残す。**

    表には `sample_id`・`speaker_id`・`speaker_group`・`label_4` が入る。
    `speaker_group` は `measure/split.py` の補正を通したもので，
    **評価はこの列で分ける**（data/README.md 取り決め 5）。

    **`with_std=True` は使わないこと**（既定が False なのはそのため）。
    映像の 2 つ目の 512 次元は**セグメント間の標準偏差**であり，セグメントが
    1 つしかない発話では必ずゼロになる。その割合はコーパスで大きく違う：

        AFEW 89.8%  CREMA-D 91.3%  MELD 64.4%  IEMOCAP 43.6%  MOSEI 17.9%

    つまりこの 512 次元は**発話の長さの代理変数**になっている（`data/README.md` 4 節
    「前身から変えた点」に前身での同じ観察がある）。実測では**セグメント数という整数 1 つだけで
    6 群を UAR 0.2822 で当てられる**（でたらめは 0.1667）。連結すれば，コーパス間
    の距離の一部が「動画の長さの違い」で説明されてしまう。
    """
    corpus, dtype = GROUP_SPEC[group]
    z = common.load_cache(kind, corpus)

    keys = FEATURE_KEYS[kind]
    X = z[keys[0]]
    if with_std and len(keys) > 1:
        X = np.concatenate([X, z[keys[1]]], axis=1)

    df = pd.DataFrame({"sample_id": z["sample_id"].astype(str)})
    man = common.load_manifest(corpus)[
        ["sample_id", "speaker_id", "dialog_type", "label_4"]]
    df = df.merge(man, on="sample_id", how="left")

    keep = (z["ok"].astype(bool) & np.isfinite(X).all(axis=1)
            & df["label_4"].notna().to_numpy())
    if dtype is not None:
        keep &= (df["dialog_type"] == dtype).to_numpy()

    out = df.loc[keep].copy()
    out["label_4"] = out["label_4"].astype(int)
    out["speaker_group"] = speaker_groups(out, corpus)
    out = out[out["speaker_group"].notna()]          # 漏れる行を落とす
    # out の index は merge 直後の行番号なので，そのまま X の行に対応する
    idx = out.index.to_numpy()
    out, X = out.reset_index(drop=True), X[idx]

    # **`sample_id` で並べ直す。**npz の行順は抽出したときのファイル走査順で
    # あり，抽出をやり直せば変わるし，モダリティごとに違いうる。行番号で標本を
    # 選ぶ箇所（K-shot の抽出など）がその順に依存する
    # ので，**ここで決定的な順序に揃える**。実例：video_IEMOCAP.npz は古い
    # マニフェスト（9,169 行）で作られ，audio は新しいもの（5,531 行）で作られた
    o = np.argsort(out["sample_id"].to_numpy(), kind="stable")
    return out.iloc[o].reset_index(drop=True), X[o]


def available(kind: str, group: str) -> bool:
    """その群・そのエンコーダの特徴が手元にあるか。抽出の途中でも動かせるように。"""
    corpus = GROUP_SPEC[group][0]
    if common.out_path(kind, corpus).is_file():
        return True
    return bool(list(common.PARTS_DIR.glob(f"{kind}_{corpus}_*.npz")))


_FUSE_STATS: dict[tuple[str, ...], list[tuple[np.ndarray, np.ndarray]]] = {}


def _fuse_stats(kinds: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """融合で使う各モダリティの平均・標準偏差を，**全群をまとめた統計**から作る。

    **2026-09-25 の修正．**以前は**群ごとに**平均 0・標準偏差 1 に揃えていたため，
    **群の間の位置と尺度の差が消えていた**（融合空間のコーパス当てが
    でたらめ 0.5 を下回る 0.099〜0.359 になって発覚した）。
    尺度を揃えたいのはモダリティの間であって群の間ではないので，
    **全群を合わせた 1 組の統計を全群に同じように当てる**。

    **2026-11-04 から，言葉を含む融合の統計に AFEW が入る**（AFEW の言葉を足したため）。
    既存の 4 群の融合特徴は列ごとの一定の倍率とずらしだけ変わり，学習データで標準化し直す
    分類器の結果は丸めの範囲でしか変わらない（`data/README.md` 7 節）。
    """
    key = tuple(kinds)
    if key in _FUSE_STATS:
        return _FUSE_STATS[key]
    gs = [g for g in GROUPS_ALL if all(available(kd, g) for kd in kinds)]
    out = []
    for kd in kinds:
        Xs = np.vstack([load_group(g, kd)[1] for g in gs])
        mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
        out.append((mu, np.where(sd > 1e-12, sd, 1.0)))
    _FUSE_STATS[key] = out
    return out


def load_fused(group: str, kinds: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """複数モダリティを連結した**融合空間**の特徴を返す（早期融合）。

    **全モダリティに存在する発話だけ**を使い，`sample_id` で突き合わせてから
    横に連結する。次元ごとに尺度が違うので標準化してからつなぐが，
    **統計は全群をまとめて 1 組だけ作り，どの群にも同じものを当てる**
    （`_fuse_stats`）。**群ごとに揃えると群の間の差が消える。**
    """
    stats = _fuse_stats(kinds)
    parts = {kd: load_group(group, kd) for kd in kinds}
    ids = set.intersection(*[set(d["sample_id"]) for d, _ in parts.values()])
    mats, ref = [], None
    for j, kd in enumerate(kinds):
        d, X = parts[kd]
        m = d["sample_id"].isin(ids).to_numpy()
        dd = d[m].reset_index(drop=True)
        o = np.argsort(dd["sample_id"].to_numpy())
        dd, Xs = dd.iloc[o].reset_index(drop=True), X[m][o]
        mu, sd = stats[j]
        mats.append((Xs - mu) / sd)
        ref = dd if ref is None else ref
    return ref, np.concatenate(mats, axis=1)


def fused_parts(kind: str) -> tuple[list[str], list[str]] | None:
    """融合名を (モダリティの一覧, 種類の一覧) にする。`fuse-audio-video` は第 1 エンコーダ，
    `fuse2-audio-video-text` は第 2 エンコーダ（HuBERT・CLIP・DeBERTa．2026-10-24，E62）。
    融合名でなければ None。"""
    for prefix, i in (("fuse-", 0), ("fuse2-", 1)):
        if kind.startswith(prefix):
            mods = kind[len(prefix):].split("-")
            return mods, [MODALITIES[m][i] for m in mods]
    return None


def load_any(group: str, kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    """`kind` が `fuse-video-audio` のような融合名なら融合空間を，
    そうでなければ単一モダリティを返す。**呼ぶ側が分岐しなくて済むようにする**
    （2026-09-25）。"""
    parts = fused_parts(kind)
    if parts:
        return load_fused(group, parts[1])
    return load_group(group, kind)


def groups_for(kind: str) -> list[str]:
    """その `kind`（単一でも融合でも）で使える群の一覧。"""
    parts = fused_parts(kind)
    if parts:
        mods, kinds = parts
        return [g for g in GROUPS_ALL
                if all(has_modality(g, m) and available(k, g)
                       for m, k in zip(mods, kinds))]
    mod = next((m for m, ks in MODALITIES.items() if kind in ks), None)
    return [g for g in GROUPS_ALL
            if (mod is None or has_modality(g, mod)) and available(kind, g)]
