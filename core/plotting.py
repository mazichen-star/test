"""统一的图表字体与字号。

论文/报告图的要求：
- 英文、数字：Times New Roman，加粗；
- 中文：宋体（SimSun / Noto Serif CJK SC），加粗；
- 坐标轴标题、图标题、图例、标注放大。

说明：Matplotlib 不会在一个 Text 对象里逐字回退字体。
因此这里按“文本里是否含中文”选择字体：
- 含中文的标题/轴标题用宋体，宋体自带 Times 风格的拉丁字形；
- 纯英文、纯数字以及数字刻度用 Times New Roman。
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.patheffects as path_effects
from matplotlib import font_manager, pyplot as plt

_NUMERIC_TICK = re.compile(r"^[\s\-−–—+.,0-9eE%×*/]+$")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")
_CONFIGURED = False
_TIMES_FONT = "Times New Roman"
_CJK_FONT = "SimSun"


def _font_available(name: str) -> bool:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        return True
    except Exception:
        return False


def _cjk_serif_font() -> str:
    """优先宋体；Linux/Cloud 上回退到内置思源宋体，避免中文方块。"""

    bundled_name = None
    font_root = Path(__file__).resolve().parent.parent / "fonts"
    for filename in ("NotoSerifSC-Regular.otf", "NotoSerifSC-VF.ttf"):
        font_path = font_root / filename
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            name = font_manager.FontProperties(fname=str(font_path)).get_name()
            if _font_available(name):
                bundled_name = name
                break
        except Exception:
            continue

    for name in (
        "SimSun",
        "Songti SC",
        "STSong",
        "Noto Serif CJK SC",
        bundled_name,
        "Noto Serif SC",
        "Source Han Serif SC",
        "AR PL UMing CN",
        # 最后的兜底：至少保证中文不是方块
        "Noto Sans SC",
        "Microsoft YaHei",
        "DejaVu Sans",
    ):
        if name and _font_available(name):
            return name
    return "DejaVu Sans"


def _times_font() -> str:
    for name in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
        if _font_available(name):
            return name
    return "DejaVu Serif"


def configure_plot_style() -> None:
    """设置全局图表样式；可重复调用。"""

    global _CONFIGURED, _TIMES_FONT, _CJK_FONT
    _CJK_FONT = _cjk_serif_font()
    _TIMES_FONT = _times_font()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                _CJK_FONT,
                "SimSun",
                "Songti SC",
                "STSong",
                "Noto Serif CJK SC",
                "Noto Serif SC",
                "Times New Roman",
                "DejaVu Serif",
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    _CONFIGURED = True


def _family_for(text: str) -> str:
    return _CJK_FONT if _CJK_CHAR.search(text or "") else _TIMES_FONT


def _style_text(
    text_obj,
    *,
    size: int | None = None,
    family: str | None = None,
    force_bold: bool = True,
) -> None:
    content = text_obj.get_text()
    if size is not None:
        text_obj.set_fontsize(size)
    if force_bold:
        text_obj.set_fontweight("bold")
    chosen = family or _family_for(content)
    text_obj.set_fontfamily(chosen)
    if chosen == _CJK_FONT:
        # SimSun 没有真正的 bold 字重，用同色描边模拟加粗。
        text_obj.set_path_effects(
            [path_effects.withStroke(linewidth=0.6, foreground=text_obj.get_color())]
        )


def style_figure(
    fig,
    *,
    tick_size: int = 13,
    label_size: int = 15,
    title_size: int = 16,
    legend_size: int = 12,
    annotation_size: int = 11,
) -> None:
    """把已生成的图统一成论文风格：数字 Times、中文宋体、文字更大更粗。"""

    if not _CONFIGURED:
        configure_plot_style()

    if getattr(fig, "_suptitle", None) is not None:
        _style_text(fig._suptitle, size=title_size + 1)

    for ax in fig.get_axes():
        _style_text(ax.title, size=title_size)

        _style_text(ax.xaxis.label, size=label_size)
        _style_text(ax.yaxis.label, size=label_size)

        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            content = tick.get_text()
            tick.set_fontsize(tick_size)
            tick.set_fontweight("bold")
            if not content:
                continue
            if _NUMERIC_TICK.fullmatch(content):
                tick.set_fontfamily(_TIMES_FONT)
            else:
                _style_text(tick, size=tick_size)

        for offset in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            if offset.get_text():
                _style_text(offset, size=max(10, tick_size - 2), family=_TIMES_FONT)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                _style_text(text, size=legend_size)

        for text in ax.texts:
            _style_text(
                text,
                size=max(text.get_fontsize(), annotation_size),
            )


def style_current_figure(**kwargs) -> None:
    style_figure(plt.gcf(), **kwargs)
