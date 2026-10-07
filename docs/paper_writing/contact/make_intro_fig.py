"""Simple Korean-labelled per-phase calibration plot for the research introduction brief."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "pdf.fonttype": 42, "font.size": 8, "axes.unicode_minus": False})

d = pd.read_csv("../../../paper/bigcomp2027/figures/fig_contrast_panel_a_per_phase.csv")
fig, ax = plt.subplots(figsize=(3.6, 2.3))
ax.plot(d.phase, d.observed_rate * 100, "-o", color="black", lw=1.6, ms=4.5, label="실제 탈락률", zorder=3)
ax.plot(d.phase, d["mean_pred__KRAFTON-tf"] * 100, "--s", color="#D55E00", lw=1.3, ms=3.5, label="기존 우승확률 모델의 예측")
ax.plot(d.phase, d["mean_pred__HGB-cut"] * 100, "-^", color="#4472C4", lw=1.3, ms=3.5, label="본 연구 모델의 예측")
ax.set_xlabel("단계 (안전구역 축소 순서)")
ax.set_ylabel("다음 단계 탈락률 (%)")
ax.set_xticks(range(1, 8))
ax.set_ylim(0, 60)
ax.grid(axis="y", color="#E5E5E5", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout(pad=0.3)
fig.savefig("intro_calibration_ko.pdf")
fig.savefig("intro_calibration_ko.png", dpi=300)
print("ok")
