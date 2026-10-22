"""统一的图表字体与字号。

论文/报告图的要求：
- 横纵坐标的数字用 Times New Roman 并加粗；
- 其他文字（轴标题、标题、图例、标注）放大并加粗；
- 中文字符继续使用项目内置的 Noto Sans SC，避免方块。
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from matplotlib import font_manager, pyplot as plt

_NUMERIC_TICK = re.compile(r"^[\s\-−–—+.,0-9eE%×*/]+$")
_CONFIGURED = False
_TIMES_FONT = "Times New Roman"


def _font_available(name: str) -> bool:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        return True
    except Exception:
        return False


def _cjk_font(preferred: str | None = None) -> str:
    candidates = [
        preferred,
        "Noto Sans SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    path = Path(__file__).resolve().parent.parent / "fonts" / "NotoSansSC-Regular.ttf"
    if path.exists():
        try:
            font_manager.fontManager.addfont(str(path))
            bundled = font_manager.FontProperties(fname=str(path)).get_name()
            candidates.insert(0, bundled)
        except Exception:
            pass
    for name in candidates:
        if name and _font_available(name):
            return name
    return "DejaVu Sans"


def _times_font() -> str:
    for name in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
        if _font_available(name):
            return name
    return "DejaVu Serif"


def configure_plot_style(cjk_font: str | None = None) -> None:
    """设置全局图表样式；可重复调用。"""

    global _CONFIGURED, _TIMES_FONT
    cjk = _cjk_font(cjk_font)
    times = _times_font()
    _TIMES_FONT = times
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                cjk,
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # 给后面的 style_figure 用
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    _CONFIGURED = True


def style_figure(
    fig,
    *,
    tick_size: int = 13,
    label_size: int = 15,
    title_size: int = 16,
    legend_size: int = 12,
    annotation_size: int = 11,
) -> None:
    """把已生成的图统一成论文风格：数字 Times、文字更大更粗。"""

    if not _CONFIGURED:
        configure_plot_style()
    times = _TIMES_FONT

    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontsize(title_size + 1)
        fig._suptitle.set_fontweight("bold")

    for ax in fig.get_axes():
        ax.title.set_fontsize(title_size)
        ax.title.set_fontweight("bold")

        for axis_label in (ax.xaxis.label, ax.yaxis.label):
            axis_label.set_fontsize(label_size)
            axis_label.set_fontweight("bold")

        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            text = tick.get_text()
            tick.set_fontsize(tick_size)
            tick.set_fontweight("bold")
            if text and _NUMERIC_TICK.fullmatch(text):
                tick.set_fontfamily(times)

        for offset in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            offset.set_fontsize(max(10, tick_size - 2))
            offset.set_fontweight("bold")
            offset.set_fontfamily(times)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_size)
                text.set_fontweight("bold")

        for text in ax.texts:
            text.set_fontsize(max(text.get_fontsize(), annotation_size))
            text.set_fontweight("bold")


def style_current_figure(**kwargs) -> None:
    style_figure(plt.gcf(), **kwargs)
