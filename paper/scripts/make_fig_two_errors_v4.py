"""ver4 labels for the two-errors figure (Fig. S2): 'prediction bias' and 'remaining error' (Japanese: 予測の偏り，残る誤り),
as defined in Sect. 3.5 of ver4. Same drawing as make_fig_two_errors.py; output fig_two_errors_v4_{ja,en}.pdf."""
import make_fig_two_errors as m

m.TXT["ja"]["b"] = ("(b) 予測の偏り（切片のずれ）", m.TXT["ja"]["b"][1].replace("答えが", "予測が"))
m.TXT["ja"]["c"] = ("(c) 残る誤り（重みの向きのずれ）", m.TXT["ja"]["c"][1])
m.TXT["en"]["b"] = ("(b) Prediction bias\n      (intercept shift)", m.TXT["en"]["b"][1].replace("the answers lean", "predictions lean"))
_save = m.save
m.save = lambda fig, name: _save(fig, name.replace("fig_two_errors_", "fig_two_errors_v4_"))

if __name__ == "__main__":
    for lang in ("ja", "en"):
        m.main(lang)
