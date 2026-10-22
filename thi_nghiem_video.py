"""Mô phỏng thí nghiệm hóa học thành video (MP4, dự phòng GIF).

Nguyên tắc: AI chỉ đọc mô tả bằng tiếng Việt và điền THAM SỐ (mô hình thí nghiệm, màu, có khí hay không,
có kết tủa hay không...). Mọi hình ảnh do bộ máy vẽ này tạo ra từ tham số, không có mã nào do AI viết.
Module không phụ thuộc Streamlit nên kiểm thử được độc lập.
"""
import io
import math
import os
import random
import re
import tempfile
import unicodedata

from PIL import Image, ImageDraw, ImageFont

W, H, S = 800, 448, 2          # kích thước video (px) và hệ số vẽ siêu mẫu
FPS = 15
BENCH = 330                    # mặt bàn thí nghiệm (tọa độ y)

MODELS = ["ong_nghiem", "khi", "chung_cat", "loc", "chuan_do", "chay"]

# ---------------------------------------------------------------- màu sắc
_RAW_COLORS = {
    "không màu": (215, 232, 242, 60), "trong suốt": (215, 232, 242, 60), "trắng": (245, 245, 245, 235),
    "đục trắng": (240, 240, 238, 215), "trắng đục": (240, 240, 238, 215), "xanh lam": (40, 110, 225, 185),
    "xanh dương": (40, 110, 225, 185), "xanh lam nhạt": (130, 190, 240, 170), "xanh nhạt": (150, 210, 235, 165),
    "xanh lục": (70, 175, 95, 185), "xanh lá": (70, 175, 95, 185), "xanh": (60, 150, 200, 180), "vàng": (240, 205, 50, 185),
    "vàng nhạt": (248, 232, 130, 160), "cam": (245, 150, 40, 190), "đỏ": (215, 45, 45, 190), "đỏ nâu": (160, 70, 40, 195),
    "nâu đỏ": (160, 70, 40, 195), "nâu": (120, 80, 45, 200), "tím": (125, 45, 160, 195), "tím đậm": (80, 25, 115, 215),
    "hồng": (255, 105, 165, 175), "đen": (35, 35, 40, 235), "xám": (130, 135, 140, 215), "bạc": (190, 196, 206, 255),
    "xanh đen": (20, 40, 90, 210), "xanh rêu": (95, 120, 70, 200), "da cam": (245, 150, 40, 190),
}
_FLAME = {
    "trắng chói": (255, 255, 235), "trắng": (255, 255, 235), "vàng": (255, 205, 70), "cam": (255, 150, 50), "đỏ cam": (255, 125, 45),
    "xanh lam nhạt": (130, 175, 255), "xanh lam": (90, 140, 255), "xanh": (110, 190, 255), "tím nhạt": (200, 150, 255),
    "đỏ": (255, 90, 60), "xanh lục": (110, 230, 140),
}


def fold(s):
    s = unicodedata.normalize("NFD", str(s or "").lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).strip()


_CF = {fold(k): v for k, v in _RAW_COLORS.items()}
_FF = {fold(k): v for k, v in _FLAME.items()}


def liq(name, default="không màu"):
    f = fold(name)
    if f in _CF:
        return _CF[f]
    for k in sorted(_CF, key=len, reverse=True):
        if k in f:
            return _CF[k]
    return _CF[fold(default)]


def flame_rgb(name):
    f = fold(name)
    if f in _FF:
        return _FF[f]
    for k in sorted(_FF, key=len, reverse=True):
        if k in f:
            return _FF[k]
    return _FF["vang"]


def color_ok(name, default="không màu"):
    f = fold(name)
    if f in _CF or any(k in f for k in _CF):
        return str(name)
    return default


def lerp(a, b, u):
    u = max(0.0, min(1.0, u))
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    return ease((t - a) / (b - a)) if b > a else (1.0 if t >= b else 0.0)


# ---------------------------------------------------------------- phông chữ
_FONTS = {}
_FONT_PATHS = {
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
           "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
}


def font(size, bold=False):
    k = (size, bold)
    if k not in _FONTS:
        f = None
        for p in _FONT_PATHS[bold]:
            try:
                f = ImageFont.truetype(p, int(size * S))
                break
            except OSError:
                continue
        _FONTS[k] = f or ImageFont.load_default()
    return _FONTS[k]


# ---------------------------------------------------------------- hình học
def arcpts(cx, cy, r, a0, a1, n=14):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def rot(pts, deg, o=(0, 0)):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(o[0] + (x - o[0]) * c - (y - o[1]) * s, o[1] + (x - o[0]) * s + (y - o[1]) * c) for x, y in pts]


def tube_outer(cx, top, w, h):
    r = w / 2
    return [(cx - r, top), (cx - r, top + h - r)] + arcpts(cx, top + h - r, r, 180, 0)[1:-1] + [(cx + r, top + h - r), (cx + r, top)]


def beaker_outer(cx, top, w, h):
    return [(cx - w / 2, top), (cx - w / 2 + 5, top + h), (cx + w / 2 - 5, top + h), (cx + w / 2, top)]


def flask_outer(cx, top, wn, hn, wb, h):
    return [(cx - wn / 2, top), (cx - wn / 2, top + hn), (cx - wb / 2, top + h), (cx + wb / 2, top + h), (cx + wn / 2, top + hn), (cx + wn / 2, top)]


def round_flask(cx, cy, R, top, nw):
    hw = nw / 2
    dy = math.sqrt(max(R * R - hw * hw, 1))
    aL = math.degrees(math.atan2(-dy, -hw)) + 360
    aR = math.degrees(math.atan2(-dy, hw))
    arc = arcpts(cx, cy, R, aL, aR, 26)
    return [(cx - hw, top)] + arc + [(cx + hw, top)]


def shrink(pts, d):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        L = math.hypot(dx, dy) or 1
        out.append((x - dx / L * d if abs(dx) > 2 else x, y - dy / L * d if abs(dy) > 2 else y))
    return out


# ---------------------------------------------------------------- khung vẽ
class Cv:
    def __init__(self, base):
        self.im = base.copy()
        self.ox = self.oy = 0
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def p(self, pts):
        return [(x * S - self.ox, y * S - self.oy) for x, y in pts]

    def poly(self, pts, fill=None, outline=None, w=2):
        P = self.p(pts)
        if fill:
            self.d.polygon(P, fill=fill)
        if outline:
            self.d.line(P + [P[0]], fill=outline, width=int(w * S), joint="curve")

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, w=2):
        a, b = self.p([(x0, y0), (x1, y1)])
        self.d.rounded_rectangle([a, b], radius=r * S, fill=fill, outline=outline, width=int(w * S))

    def ell(self, cx, cy, rx, ry, fill=None, outline=None, w=2):
        a, b = self.p([(cx - rx, cy - ry), (cx + rx, cy + ry)])
        self.d.ellipse([a, b], fill=fill, outline=outline, width=int(w * S))

    def line(self, pts, fill, w=2):
        self.d.line(self.p(pts), fill=fill, width=int(w * S), joint="curve")

    def text(self, x, y, s, size=16, fill=(255, 255, 255), bold=False, anchor="la"):
        (px, py), = self.p([(x, y)])
        self.d.text((px, py), s, font=font(size, bold), fill=fill, anchor=anchor)

    def tlen(self, s, size, bold=False):
        return font(size, bold).getlength(s) / S

    def wrap(self, s, size, maxw, bold=False):
        lines, cur = [], ""
        for w in str(s).split():
            test = (cur + " " + w).strip()
            if self.tlen(test, size, bold) <= maxw or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def clip(self, poly, fn):
        P = [(x * S - self.ox, y * S - self.oy) for x, y in poly]
        x0 = max(0, int(min(p[0] for p in P)) - 2)
        y0 = max(0, int(min(p[1] for p in P)) - 2)
        x1 = min(self.im.size[0], int(max(p[0] for p in P)) + 3)
        y1 = min(self.im.size[1], int(max(p[1] for p in P)) + 3)
        if x1 <= x0 or y1 <= y0:
            return
        sub = Cv.__new__(Cv)
        sub.im = self.im.crop((x0, y0, x1, y1))
        sub.ox, sub.oy = self.ox + x0, self.oy + y0
        sub.d = ImageDraw.Draw(sub.im, "RGBA")
        fn(sub)
        mask = Image.new("L", sub.im.size, 0)
        ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in P], fill=255)
        self.im.paste(sub.im, (x0, y0), mask)


_BG = None


def background():
    global _BG
    if _BG is None:
        im = Image.new("RGB", (W * S, H * S))
        d = ImageDraw.Draw(im)
        for y in range(H * S):
            u = y / (H * S)
            if y < BENCH * S:
                c = lerp((236, 243, 248), (214, 227, 238), u / (BENCH / H))
            else:
                v = (y - BENCH * S) / ((H - BENCH) * S)
                c = lerp((178, 138, 98), (128, 92, 60), v)
            d.line([(0, y), (W * S, y)], fill=c)
        d.rectangle([0, BENCH * S, W * S, (BENCH + 5) * S], fill=(196, 160, 120))
        d.line([(0, BENCH * S), (W * S, BENCH * S)], fill=(110, 78, 50), width=2 * S)
        _BG = im
    return _BG


# ---------------------------------------------------------------- vật dụng chung
GLASS_LINE = (110, 140, 165, 255)


def glass(cv, poly, hl=True):
    cv.poly(poly, fill=(255, 255, 255, 38), outline=GLASS_LINE, w=2)
    if hl and len(poly) > 3:
        x0 = min(p[0] for p in poly)
        ys = [p[1] for p in poly]
        cv.line([(x0 + 6, min(ys) + 14), (x0 + 6, min(ys) + (max(ys) - min(ys)) * 0.55)], (255, 255, 255, 170), 3)


def liquid_in(cv, inner, y_top, color, bottom=None):
    r, g, b, a = color
    ys = [p[1] for p in inner]
    xs = [p[0] for p in inner]
    yb = bottom if bottom is not None else max(ys) + 2
    cv.clip(inner, lambda c: c.d.rectangle(
        [(min(xs) - 2) * S - c.ox, y_top * S - c.oy, (max(xs) + 2) * S - c.ox, (yb + 4) * S - c.oy], fill=(r, g, b, a)))
    # mặt thoáng
    cv.clip(inner, lambda c: c.d.rectangle([(min(xs) - 2) * S - c.ox, y_top * S - c.oy, (max(xs) + 2) * S - c.ox, (y_top + 2.5) * S - c.oy], fill=(255, 255, 255, 110)))


def bubbles(cv, area, t, n, speed=0.55, color=(255, 255, 255), seed=1, rmax=4.2, alpha=190):
    """area = (x0, y_top, x1, y_bottom). Bọt khí nổi lên từ đáy tới mặt thoáng."""
    x0, y0, x1, y1 = area
    rnd = random.Random(seed)
    for i in range(int(n)):
        ph = rnd.random()
        sx = rnd.uniform(x0, x1)
        r = rnd.uniform(1.8, rmax)
        wob = rnd.uniform(1.5, 4)
        u = (t * speed + ph) % 1.0
        y = y1 - (y1 - y0) * u
        x = sx + math.sin(t * 4 + i) * wob
        cv.ell(x, y, r, r, fill=(color[0], color[1], color[2], 40), outline=(color[0], color[1], color[2], alpha), w=1.3)


def lamp(cv, x, on=1.0, t=0.0, y_base=BENCH):
    cv.poly([(x - 22, y_base), (x - 30, y_base - 14), (x - 20, y_base - 32), (x + 20, y_base - 32), (x + 30, y_base - 14), (x + 22, y_base)],
            fill=(214, 232, 240, 210), outline=GLASS_LINE, w=2)
    cv.rrect(x - 17, y_base - 24, x + 17, y_base - 4, 4, fill=(235, 245, 150, 110))
    cv.rrect(x - 8, y_base - 42, x + 8, y_base - 32, 2, fill=(160, 165, 170, 255))
    cv.line([(x, y_base - 42), (x, y_base - 48)], (60, 50, 40, 255), 3)
    if on > 0.02:
        flame(cv, x, y_base - 48, 46 * on, t, (255, 190, 70), glow=True)


def flame(cv, x, y_base, h, t, rgb=(255, 190, 70), glow=False, w=None):
    if h <= 1:
        return
    w = w or h * 0.34
    fl = 1 + 0.07 * math.sin(t * 17) + 0.04 * math.sin(t * 31 + 1)
    hh = h * fl
    if glow:
        for k, al in ((3.2, 18), (2.2, 28), (1.5, 40)):
            cv.ell(x, y_base - hh * 0.45, w * k, hh * 0.65, fill=(rgb[0], rgb[1], rgb[2], al))
    pts = [(x, y_base - hh), (x + w * 0.55, y_base - hh * 0.55), (x + w, y_base - hh * 0.2), (x + w * 0.7, y_base), (x - w * 0.7, y_base),
           (x - w, y_base - hh * 0.2), (x - w * 0.55, y_base - hh * 0.55)]
    cv.poly(pts, fill=(rgb[0], rgb[1], rgb[2], 215))
    inner = [(px * 0.55 + x * 0.45, y_base - (y_base - py) * 0.7) for px, py in pts]
    cv.poly(inner, fill=(255, 250, 215, 230))


def stand(cv, x, top, arm_y=None, arm_to=None, base_y=BENCH):
    cv.rrect(x - 52, base_y - 8, x + 52, base_y + 2, 3, fill=(70, 78, 90, 255))
    cv.rrect(x - 4, top, x + 4, base_y - 6, 2, fill=(110, 120, 132, 255))
    if arm_y is not None and arm_to is not None:
        cv.line([(x, arm_y), arm_to], (98, 108, 120, 255), 5)
        cv.ell(arm_to[0], arm_to[1], 7, 7, fill=(70, 78, 90, 255))


def tag(cv, x, y, text, color=(30, 90, 140), align="l"):
    wd = cv.tlen(text, 13, True) + 16
    x0 = x if align == "l" else x - wd
    cv.rrect(x0, y, x0 + wd, y + 24, 12, fill=(255, 255, 255, 225), outline=color + (255,), w=1.5)
    cv.text(x0 + 8, y + 12, text, 13, fill=color, bold=True, anchor="lm")


def dropper(cv, x, y_tip, color):
    cv.poly([(x - 3, y_tip), (x + 3, y_tip), (x + 6, y_tip - 40), (x - 6, y_tip - 40)], fill=(230, 240, 246, 230), outline=GLASS_LINE, w=1.5)
    cv.ell(x, y_tip - 54, 11, 16, fill=(60, 60, 70, 255))
    cv.rrect(x - 8, y_tip - 44, x + 8, y_tip - 38, 2, fill=(110, 110, 120, 255))
    r, g, b, a = color
    cv.ell(x, y_tip - 8, 2.5, 6, fill=(r, g, b, min(255, a + 40)))


def drop_fall(cv, x, y0, y1, t, period, color, t0=0.0, t1=1e9, n_vis=1):
    """Giọt chất lỏng rơi lặp lại từ y0 tới y1."""
    if not (t0 <= t <= t1):
        return
    r, g, b, a = color
    u = ((t - t0) / period) % 1.0
    if u < 0.25:
        cv.ell(x, y0, 3.2 * (u / 0.25) + 0.5, 4.2 * (u / 0.25) + 0.8, fill=(r, g, b, min(255, a + 50)))
    else:
        v = (u - 0.25) / 0.75
        y = y0 + (y1 - y0) * v * v
        cv.ell(x, y, 3, 4.6, fill=(r, g, b, min(255, a + 50)))


def smoke(cv, x, y, t, n=10, color=(235, 235, 235), seed=3, spread=30, rise=120, alpha=120):
    rnd = random.Random(seed)
    for i in range(n):
        ph = rnd.random()
        u = (t * 0.35 + ph) % 1.0
        sx = x + rnd.uniform(-8, 8) + math.sin(t * 2 + i) * spread * u
        sy = y - rise * u
        rr = 8 + 22 * u
        cv.ell(sx, sy, rr, rr * 0.85, fill=(color[0], color[1], color[2], int(alpha * (1 - u) ** 1.2)))


# ---------------------------------------------------------------- tiêu đề, phụ đề, kết quả
def title_bar(cv, text):
    wd = min(W - 24, cv.tlen(text, 18, True) + 28)
    cv.rrect(12, 10, 12 + wd, 42, 14, fill=(255, 255, 255, 225), outline=(14, 143, 110, 255), w=2)
    cv.text(26, 26, text if cv.tlen(text, 18, True) < W - 56 else text[:60] + "…", 18, fill=(14, 80, 70), bold=True, anchor="lm")


def caption_bar(cv, text, idx, n):
    cv.rrect(0, 394, W, H, 0, fill=(14, 32, 52, 232))
    cv.ell(34, 421, 17, 17, fill=(20, 184, 166, 255))
    cv.text(34, 421, str(idx + 1), 17, fill=(255, 255, 255), bold=True, anchor="mm")
    lines = cv.wrap(text, 17, W - 100)[:2]
    for i, ln in enumerate(lines):
        cv.text(64, 421 + (i - (len(lines) - 1) / 2) * 21, ln, 17, fill=(255, 255, 255), anchor="lm")
    cv.text(W - 14, 405, f"Bước {idx + 1}/{n}", 12, fill=(160, 220, 210), anchor="ra")


def result_card(cv, sc, alpha):
    if alpha <= 0.01:
        return
    a = int(235 * alpha)
    cv.rrect(18, 54, W - 18, 384, 18, fill=(255, 255, 255, a), outline=(14, 143, 110, int(255 * alpha)), w=2)
    y = 72
    cv.text(38, y, "Hiện tượng quan sát được", 20, fill=(14, 100, 80, int(255 * alpha)), bold=True)
    y += 34
    for ln in sc["hien_tuong"][:5]:
        for j, row in enumerate(cv.wrap(ln, 16, W - 120)):
            if j == 0:
                cv.ell(46, y + 9, 4, 4, fill=(20, 184, 166, int(255 * alpha)))
            cv.text(60, y, row, 16, fill=(30, 45, 55, int(255 * alpha)))
            y += 22
        y += 4
    eq = (sc.get("phuong_trinh") or {}).get("text") or ""
    if eq and y < 330:
        y += 4
        cv.text(38, y, "Phương trình", 18, fill=(14, 100, 80, int(255 * alpha)), bold=True)
        y += 28
        for row in cv.wrap(eq, 17, W - 80)[:2]:
            cv.text(38, y, row, 17, fill=(30, 45, 55, int(255 * alpha)), bold=True)
            y += 24


# ---------------------------------------------------------------- khung thời gian
def phases(T):
    """Bốn giai đoạn: chuẩn bị, thêm chất, phản ứng, quan sát; sau đó là thẻ kết quả."""
    return [0.0, 0.20 * T, 0.40 * T, 0.78 * T, T]


# ================================================================ MÔ HÌNH 1: ống nghiệm / cốc
def draw_ong_nghiem(cv, t, sc, T):
    P = phases(T)
    beaker = sc["vat"] == "coc"
    heat = sc["dun_nong"]
    cx = 400
    c0 = liq(sc["chat_dau"]["mau"])
    c_end = liq(sc.get("mau_sau") or sc["chat_dau"]["mau"])
    add_liq = sc["them"]["kieu"] == "long"
    add_col = liq(sc["them"]["mau"])
    # hình học vật chứa
    if beaker:
        w, h, top = 170, 140, BENCH - 140
        outer = beaker_outer(cx, top, w, h)
        inner = beaker_outer(cx, top + 3, w - 10, h - 5)
        bottom = BENCH - 2
    elif heat:
        w, h = 46, 175
        top = 70
        outer = tube_outer(cx, top, w, h)
        inner = tube_outer(cx, top + 2, w - 7, h - 3)
        bottom = top + h - 3
    else:
        w, h, top = 48, 205, BENCH - 205
        outer = tube_outer(cx, top, w, h)
        inner = tube_outer(cx, top + 2, w - 7, h - 3)
        bottom = BENCH - 3
    depth = bottom - top
    lv0 = bottom - depth * (0.34 if not beaker else 0.30)
    lv1 = lv0 - depth * (0.10 if add_liq else 0.03)
    # dụng cụ phụ trợ phía sau
    if heat and not beaker:
        stand(cv, cx - 150, 90, arm_y=120, arm_to=(cx - 18, 120))
        cv.line([(cx - 150, 120), (cx - 22, 120)], (98, 108, 120, 255), 5)
        cv.rrect(cx - 30, 108, cx + 30, 130, 5, fill=(150, 110, 70, 255))
    elif not beaker:
        cv.rrect(cx - 78, 282, cx + 78, 292, 3, fill=(176, 128, 82, 255))
        cv.rrect(cx - 78, 292, cx - 68, BENCH, 2, fill=(150, 108, 68, 255))
        cv.rrect(cx + 68, 292, cx + 78, BENCH, 2, fill=(150, 108, 68, 255))
    # mức chất lỏng theo thời gian
    k_add = seg(t, P[1], P[1] + 0.8 * (P[2] - P[1]))
    level = lv0 - (lv0 - lv1) * (k_add if add_liq else seg(t, P[1] + 0.3 * (P[2] - P[1]), P[2]))
    # màu theo thời gian
    k_col = seg(t, P[2] + 0.04 * T, P[3] - 0.06 * T)
    col = lerp(c0, c_end, k_col)
    liquid_in(cv, inner, level, col)
    # kết tủa
    ppt = sc.get("ket_tua")
    if ppt:
        pc = liq(ppt)
        pc = (pc[0], pc[1], pc[2], 235)
        tt0, tt1 = P[2], P[3] - 0.04 * T
        rnd = random.Random(7)
        for i in range(46):
            s0 = tt0 + rnd.random() * (tt1 - tt0) * 0.8
            if t < s0:
                continue
            x = cx + rnd.uniform(-w * 0.3, w * 0.3)
            ys = level + 6 + rnd.random() * (bottom - level) * 0.4
            fall = ease((t - s0) / (0.28 * T))
            y = ys + (bottom - 6 - ys) * fall
            cv.clip(inner, lambda c, x=x, y=y: c.ell(x, y, 2.6, 2.6, fill=pc))
        layer = seg(t, tt0 + 0.1 * T, tt1 + 0.1 * T) * 16
        if layer > 0.5:
            cv.clip(inner, lambda c: c.ell(cx, bottom - layer * 0.4, w * 0.5, layer, fill=pc))
    # chất thêm vào
    px, py_end = cx, bottom - 14
    if sc["them"]["kieu"] == "ran":
        sol = liq(sc["them"]["mau"])
        sol = (sol[0], sol[1], sol[2], 255)
        k_in = seg(t, P[1], P[1] + 0.65 * (P[2] - P[1]))
        if t < P[1]:
            ysol, show = None, False
        else:
            show = True
            ystart = 60
            if sc["bay"]:
                ysol = ystart + (level - 4 - ystart) * k_in
            else:
                ysol = ystart + (py_end - ystart) * k_in * k_in
            if k_in < 1 and t < P[1] + 0.5 * (P[2] - P[1]):
                # kẹp đưa chất rắn vào
                cv.line([(px + 6, ysol - 40), (px + 6, ysol - 8)], (120, 125, 135, 255), 3)
                cv.line([(px - 6, ysol - 40), (px - 6, ysol - 8)], (120, 125, 135, 255), 3)
        if show:
            shrink_k = seg(t, P[2] + 0.04 * T, P[3]) if sc["ran_tan"] else 0
            size = 11 * (1 - 0.92 * shrink_k) if not sc["bay"] else 9 * (1 - 0.85 * shrink_k)
            sx = px
            if sc["bay"] and k_in >= 1:
                sx = px + math.sin(t * 2.3) * (w * 0.32) * seg(t, P[2], P[2] + 1.2)
                ysol = level - 2 + math.sin(t * 7) * 1.2
            coat = sc.get("bam")
            solc = lerp(sol, (liq(coat)[0], liq(coat)[1], liq(coat)[2], 255), seg(t, P[2] + 0.06 * T, P[3] - 0.04 * T)) if coat else sol
            if size > 0.8:
                if beaker or True:
                    cv.clip(inner + [(cx, 0)] if False else [(cx - w, 0), (cx + w, 0), (cx + w, BENCH + 5), (cx - w, BENCH + 5)],
                            lambda c: c.rrect(sx - size, ysol - size * 0.8, sx + size, ysol + size * 0.8, 3, fill=solc, outline=(60, 60, 60, 160), w=1))
    else:
        if P[1] <= t <= P[2] + 0.1 * T:
            dropper(cv, cx, 74, add_col)
            drop_fall(cv, cx, 74, level - 2, t, 0.42, add_col, t0=P[1], t1=P[2] - 0.2)
    # bọt khí
    if sc["khi"]:
        gk = seg(t, P[2], P[2] + 0.1 * T) * (1 - 0.65 * seg(t, P[3] - 0.15 * T, P[3] + 0.1 * T))
        if gk > 0.02:
            gc = liq(sc["khi_mau"])[:3] if fold(sc["khi_mau"]) not in ("khong mau", "trong suot") else (255, 255, 255)
            cv.clip(inner, lambda c: bubbles(c, (cx - w * 0.3, level + 2, cx + w * 0.3, bottom - 6), t, 14 * gk, speed=0.7, color=gc, seed=11))
            if gk > 0.5:
                for i in range(6):
                    fx = cx + (i - 2.5) * (w * 0.11)
                    cv.ell(fx, level - 1, 3.2, 2.4, fill=(255, 255, 255, 140), outline=(255, 255, 255, 220), w=1)
            if sc["khi_mau"] and fold(sc["khi_mau"]) not in ("khong mau", "trong suot"):
                for i in range(7):
                    u = ((t * 0.5 + i / 7) % 1.0)
                    r2 = liq(sc["khi_mau"])
                    cv.ell(cx + math.sin(t * 2 + i) * 14, level - 8 - 70 * u, 8 + 8 * u, 6 + 6 * u, fill=(r2[0], r2[1], r2[2], int(70 * (1 - u) * gk)))
    # sủi bọt khi dun nóng
    if heat:
        lamp_x = cx
        lamp(cv, lamp_x, on=seg(t, P[1] + 0.05 * T, P[1] + 0.2 * T), t=t, y_base=BENCH)
        if t > P[2]:
            cv.clip(inner, lambda c: bubbles(c, (cx - w * 0.28, level + 2, cx + w * 0.28, bottom - 4), t, 6, speed=0.5, seed=5, rmax=2.6, alpha=150))
    # tạo kính sau cùng
    glass(cv, outer)
    # nhãn
    tag(cv, 28, 62, f"{'Cốc' if beaker else 'Ống nghiệm'}: {sc['chat_dau']['ten']}", (30, 90, 140))
    if sc["them"]["ten"]:
        tag(cv, 28, 92, f"Thêm vào: {sc['them']['ten']}", (170, 80, 30))


# ================================================================ MÔ HÌNH 2: điều chế và thu khí
def _stopper(cv, x, y, ang):
    pts = rot([(x - 9, y - 14), (x + 9, y - 14), (x + 12, y + 6), (x - 12, y + 6)], ang, (x, y))
    cv.poly(pts, fill=(160, 112, 72, 255), outline=(110, 74, 44, 255), w=1.5)


def draw_khi(cv, t, sc, T):
    P = phases(T)
    water = sc["thu"] == "day_nuoc"
    gas = sc["khi"]
    gcol = liq(gas["mau"])
    gtint = (gcol[0], gcol[1], gcol[2], 85 if fold(gas["mau"]) in ("khong mau", "trong suot") else 120)
    if fold(gas["mau"]) in ("khong mau", "trong suot"):
        gtint = (190, 215, 240, 90)
    heat_k = seg(t, P[1], P[1] + 0.18 * T)
    prod = seg(t, P[2] - 0.02 * T, P[3] - 0.22 * T)          # lượng khí đã sinh ra (0..1)
    # ---- nguồn khí
    if sc["cach"] == "dun_ran":
        M = (318, 232)
        ang = 106
        L = 168
        tube = rot(tube_outer(M[0], M[1], 44, L), ang, M)
        inner = rot(tube_outer(M[0], M[1] + 2, 37, L - 3), ang, M)
        stand(cv, 118, 96, arm_y=124, arm_to=(M[0] - 36, 124))
        cv.line([(118, 124), (M[0] - 36, 124), (M[0] - 36, 212)], (98, 108, 120, 255), 5)
        s0 = sc["chat_ran"]
        c_a = liq(s0["mau"])
        c_b = liq(s0.get("mau_sau") or s0["mau"])
        pc = lerp(c_a, c_b, seg(t, P[2], P[3] - 0.1 * T))
        powder = rot([(M[0] - 15, M[1] + 94), (M[0] + 15, M[1] + 94), (M[0] + 15, M[1] + L - 10), (M[0] - 15, M[1] + L - 10)], ang, M)
        cv.clip(inner, lambda c: c.poly(powder, fill=(pc[0], pc[1], pc[2], 255)))
        glass(cv, tube, hl=False)
        _stopper(cv, M[0] + 2, M[1] + 2, ang)
        lamp(cv, 214, on=heat_k, t=t)
        start = (M[0] + 12, M[1] + 4)
    else:
        fx, fy = 215, BENCH
        outer = flask_outer(fx, fy - 135, 34, 52, 128, 135)
        inn = flask_outer(fx, fy - 133, 28, 50, 120, 131)
        c0 = liq(sc["chat_long"]["mau"])
        lvl = fy - 52
        liquid_in(cv, inn, lvl, c0)
        sol = liq(sc["chat_ran"]["mau"])
        for i in range(4):
            cv.clip(inn, lambda c, i=i: c.rrect(fx - 34 + i * 17, fy - 12, fx - 22 + i * 17, fy - 3, 3, fill=(sol[0], sol[1], sol[2], 255), outline=(60, 60, 60, 120), w=1))
        if prod > 0.02:
            cv.clip(inn, lambda c: bubbles(c, (fx - 40, lvl + 2, fx + 40, fy - 8), t, 12 * prod, speed=0.8, seed=21))
        glass(cv, outer)
        # phễu thêm chất lỏng
        cv.poly([(fx - 22, 70), (fx + 22, 70), (fx + 4, 112), (fx - 4, 112)], fill=(240, 246, 250, 160), outline=GLASS_LINE, w=1.5)
        cv.rrect(fx - 3, 112, fx + 3, fy - 62, 1, fill=(220, 232, 240, 200), outline=GLASS_LINE, w=1)
        liquid_in(cv, [(fx - 20, 72), (fx + 20, 72), (fx + 3, 110), (fx - 3, 110)], 84, c0)
        if P[1] <= t <= P[3]:
            drop_fall(cv, fx, 116, lvl - 4, t, 0.5, c0, t0=P[1], t1=P[3] - 0.1 * T)
        _stopper(cv, fx, fy - 135 + 5, 0)
        start = (fx + 14, fy - 128)
    # ---- ống dẫn khí
    if water:
        tx0, tx1, tw_top, tw_y = 480, 770, 270, 330
        wl = 292
        path = [start, (start[0] + 40, start[1] + 6), (540, start[1] + 8), (540, 314), (620, 318)]
    elif sc["thu"] == "day_kk_up":
        path = [start, (start[0] + 40, start[1] + 6), (610, start[1] + 8), (610, 318)]
    else:
        path = [start, (start[0] + 40, start[1] + 6), (610, start[1] + 8), (610, 224)]
    cv.line(path, (125, 150, 175, 255), 7)
    cv.line(path, (225, 240, 248, 230), 3.4)
    # ---- thu khí
    if water:
        trough = [(480, 276), (770, 276), (770, BENCH), (480, BENCH)]
        cv.poly(trough, fill=(200, 225, 240, 80), outline=GLASS_LINE, w=2)
        cv.clip(trough, lambda c: c.rrect(481, wl, 769, BENCH, 0, fill=(70, 140, 210, 120)))
        cv.rrect(480, wl - 1, 770, wl + 2, 0, fill=(255, 255, 255, 120))
        ct_top, ct_mouth, cxx = 168, 312, 620
        out_t = tube_outer(cxx, ct_top, 50, 150)
        out_t = [(x, y) for x, y in out_t]
        # ống úp: dùng ống thẳng, miệng ở dưới
        shape = [(cxx - 25, ct_top + 25), (cxx - 25, ct_mouth), (cxx + 25, ct_mouth), (cxx + 25, ct_top + 25)] + arcpts(cxx, ct_top + 25, 25, 0, -180)[1:-1][::-1][::-1]
        shape = [(cxx - 25, ct_mouth), (cxx - 25, ct_top + 25)] + arcpts(cxx, ct_top + 25, 25, 180, 360)[1:-1] + [(cxx + 25, ct_top + 25), (cxx + 25, ct_mouth)]
        g_lvl = ct_top + 6 + (wl + 4 - ct_top - 6) * prod
        cv.clip(shape, lambda c: c.rrect(cxx - 26, ct_top, cxx + 26, g_lvl, 0, fill=gtint))
        cv.clip(shape, lambda c: c.rrect(cxx - 26, g_lvl, cxx + 26, ct_mouth + 2, 0, fill=(70, 140, 210, 120)))
        glass(cv, shape)
        if prod > 0.02:
            cv.text(cxx, ct_top + 40, gas["ten"], 15, fill=(30, 60, 90, 230), bold=True, anchor="mm")
        endx, endy = 620, 316
        if 0.0 < prod < 0.995 or t < P[3]:
            cv.clip(shape + [(cxx - 25, ct_mouth + 30), (cxx + 25, ct_mouth + 30)],
                    lambda c: bubbles(c, (endx - 6, g_lvl + 2, endx + 6, endy), t, 7 * (prod > 0.01), speed=0.9, seed=31, rmax=3.2))
    else:
        jx = 610
        if sc["thu"] == "day_kk_up":
            jar = [(jx - 38, 190), (jx - 38, BENCH), (jx + 38, BENCH), (jx + 38, 190)]
            g_top = BENCH - (BENCH - 192) * prod
            cv.clip(jar, lambda c: c.rrect(jx - 40, g_top, jx + 40, BENCH + 2, 0, fill=gtint))
            glass(cv, jar)
            cv.text(jx, BENCH - 20, gas["ten"], 15, fill=(30, 60, 90, 230), bold=True, anchor="mm") if prod > 0.1 else None
        else:
            jar = [(jx - 38, 224), (jx - 38, 100), (jx + 38, 100), (jx + 38, 224)]
            g_bot = 100 + (224 - 100) * prod
            cv.clip(jar, lambda c: c.rrect(jx - 40, 100, jx + 40, g_bot, 0, fill=gtint))
            glass(cv, jar)
            cv.text(jx, 130, gas["ten"], 15, fill=(30, 60, 90, 230), bold=True, anchor="mm") if prod > 0.1 else None
    # ---- thử khí
    test = sc["thu_thu"]
    k_test = seg(t, P[3] - 0.22 * T, P[3] - 0.02 * T)
    if test == "que_dom" and k_test > 0.01:
        tx = 430 if water else 440
        up = tube_outer(tx, 150, 52, 180)
        cv.poly(up, fill=(255, 255, 255, 28))
        cv.clip(up, lambda c: c.rrect(tx - 30, 150, tx + 30, BENCH, 0, fill=gtint))
        glass(cv, up)
        sy = 90 + 120 * k_test
        cv.line([(tx + 30, sy - 70), (tx + 4, sy)], (190, 150, 100, 255), 5)
        flare = seg(k_test, 0.6, 0.9)
        cv.ell(tx + 4, sy, 5, 5, fill=(255, 120, 40, 255))
        if flare > 0.02:
            flame(cv, tx + 2, sy + 4, 56 * flare, t, (255, 235, 140), glow=True, w=16 * flare)
        tag(cv, tx - 60, 118, "Que đóm bùng cháy" if flare > 0.3 else "Que đóm còn tàn đỏ", (180, 70, 30))
    elif test == "tieng_no_nho" and k_test > 0.01:
        tx = 430
        up = tube_outer(tx, 150, 52, 180)
        glass(cv, up)
        cv.clip(up, lambda c: c.rrect(tx - 30, 150, tx + 30, BENCH, 0, fill=gtint))
        fl = seg(k_test, 0.5, 0.8) * (1 - seg(k_test, 0.85, 1.0))
        if fl > 0.02:
            flame(cv, tx, 160, 40 * fl, t, (255, 210, 90), glow=True)
        tag(cv, tx - 60, 118, "Cháy kèm tiếng nổ nhỏ" if fl > 0.2 else "Đưa lửa lại gần miệng ống", (180, 70, 30))
    tag(cv, 28, 62, f"Nguồn khí: {(sc['chat_ran'] or {}).get('ten', '')}" + (f" + {sc['chat_long']['ten']}" if sc["cach"] == "ran_long" else ""), (30, 90, 140))
    tag(cv, 28, 92, f"Khí thu được: {gas['ten']}", (170, 80, 30))
    tag(cv, 28, 122, {"day_nuoc": "Thu bằng cách đẩy nước", "day_kk_up": "Đẩy không khí, bình để ngửa", "day_kk_ngua": "Đẩy không khí, bình úp"}[sc["thu"]], (14, 110, 90))


# ================================================================ MÔ HÌNH 3: chưng cất
def draw_chung_cat(cv, t, sc, T):
    P = phases(T)
    c0 = liq(sc["chat_long"]["mau"])
    cd = liq(sc["chat_cat"]["mau"], default="không màu")
    bp = sc["nhiet_do_soi"]
    heat_k = seg(t, P[1], P[1] + 0.18 * T)
    temp = 25 + (bp - 25) * seg(t, P[1] + 0.05 * T, P[2] + 0.12 * T)
    boil = seg(t, P[2], P[2] + 0.08 * T)
    cx, cy, R = 190, 212, 56
    # chân kiềng và lưới
    cv.line([(cx - 62, 286), (cx - 74, BENCH)], (80, 88, 100, 255), 4)
    cv.line([(cx + 62, 286), (cx + 74, BENCH)], (80, 88, 100, 255), 4)
    cv.rrect(cx - 66, 282, cx + 66, 288, 2, fill=(130, 138, 150, 255))
    flask = round_flask(cx, cy, R, 112, 36)
    inn = round_flask(cx, cy, R - 4, 114, 30)
    k_vol = seg(t, P[2] + 0.05 * T, P[3])
    level = cy - 4 + 14 * k_vol
    liquid_in(cv, inn, level, c0)
    if boil > 0.05:
        cv.clip(inn, lambda c: bubbles(c, (cx - 34, level + 2, cx + 34, cy + R - 6), t, 12 * boil, speed=0.9, seed=41, rmax=3.2))
    glass(cv, flask)
    cv.rrect(cx - 20, 106, cx + 20, 118, 3, fill=(160, 112, 72, 255))
    # nhiệt kế
    cv.rrect(cx - 3, 72, cx + 3, 236, 2, fill=(235, 242, 246, 200), outline=GLASS_LINE, w=1)
    hgt = 236 - (236 - 120) * ((temp - 20) / (max(bp, 40) + 20 - 20))
    cv.rrect(cx - 1.5, hgt, cx + 1.5, 236, 1, fill=(220, 40, 40, 255))
    cv.ell(cx, 238, 5, 5, fill=(220, 40, 40, 255))
    tag(cv, cx + 14, 74, f"t = {temp:.0f} °C", (190, 40, 40))
    lamp(cv, cx, on=heat_k, t=t)
    # ống sinh hàn: nghiêng xuống phải
    A, B = (cx + 20, 150), (520, 262)
    ang = math.degrees(math.atan2(B[1] - A[1], B[0] - A[0]))
    Ln = math.hypot(B[0] - A[0], B[1] - A[1])
    jacket = rot([(A[0], A[1] - 16), (A[0] + Ln - 20, A[1] - 16), (A[0] + Ln - 20, A[1] + 16), (A[0], A[1] + 16)], ang, A)
    cv.poly(jacket, fill=(150, 205, 240, 110), outline=GLASS_LINE, w=2)
    inner_t = rot([(A[0] - 20, A[1] - 4), (A[0] + Ln, A[1] - 4), (A[0] + Ln, A[1] + 4), (A[0] - 20, A[1] + 4)], ang, A)
    cv.poly(inner_t, fill=(235, 245, 250, 200), outline=GLASS_LINE, w=1.5)
    for i in range(6):
        u = (i / 6 + t * 0.4) % 1
        px, py = A[0] + (Ln - 30) * (1 - u) + 6, A[1] - 26
        (qx, qy), = rot([(A[0] + (Ln - 40) * (1 - u) + 10, A[1] - 24)], ang, A)
        cv.line([(qx, qy), (qx + 9 * math.cos(math.radians(ang + 180)), qy + 9 * math.sin(math.radians(ang + 180)))], (60, 140, 215, 235), 2.4)
    tag(cv, A[0] + 70, 112, "Nước làm lạnh", (40, 110, 190))
    # hơi trên đường ống
    if boil > 0.1:
        for i in range(5):
            u = (i / 5 + t * 0.45) % 1
            (qx, qy), = rot([(A[0] + (Ln - 60) * u, A[1])], ang, A)
            cv.ell(qx, qy, 4, 3, fill=(255, 255, 255, int(200 * boil * (1 - u * 0.6))))
    # bình hứng
    rx, ry = 610, BENCH
    rec = flask_outer(rx, ry - 110, 30, 38, 110, 110)
    rin = flask_outer(rx, ry - 108, 24, 36, 102, 106)
    rl = ry - 4 - 38 * seg(t, P[2] + 0.1 * T, P[3] + 0.02 * T)
    if rl < ry - 4:
        liquid_in(cv, rin, rl, cd)
    glass(cv, rec)
    drip_on = seg(t, P[2] + 0.1 * T, P[2] + 0.2 * T) * (1 - seg(t, P[3] + 0.05 * T, P[3] + 0.15 * T))
    if drip_on > 0.3:
        drop_fall(cv, B[0] + 6, B[1] + 6, min(rl, ry - 10) - 2 if rl < ry - 4 else ry - 12, t, 0.55, cd if fold(sc["chat_cat"]["mau"]) not in ("khong mau",) else (170, 200, 230, 190), t0=0)
    tag(cv, 28, 62, f"Bình cất: {sc['chat_long']['ten']}", (30, 90, 140))
    tag(cv, 28, 92, f"Chất cất được: {sc['chat_cat']['ten']}", (170, 80, 30))


# ================================================================ MÔ HÌNH 4: lọc
def draw_loc(cv, t, sc, T):
    P = phases(T)
    mix = sc["hon_hop"]
    c_mix = liq(mix["mau_dd"])
    c_res = liq(mix["mau_ran"])
    c_res = (c_res[0], c_res[1], c_res[2], 255)
    c_fil = liq(sc["dich_loc"]["mau"])
    fx = 330
    stand(cv, 190, 80, arm_y=150, arm_to=(fx - 60, 150))
    cv.line([(190, 150), (fx - 58, 150)], (98, 108, 120, 255), 5)
    cv.ell(fx, 150, 66, 8, outline=(98, 108, 120, 255), w=4)
    pour = seg(t, P[1], P[1] + 0.05 * T) * (1 - seg(t, P[2] + 0.16 * T, P[2] + 0.22 * T))
    flow = seg(t, P[1] + 0.05 * T, P[2] + 0.2 * T)         # lượng dung dịch đã qua giấy lọc
    # cốc hứng
    bw, bh = 130, 120
    btop = BENCH - bh
    bo = beaker_outer(fx, btop, bw, bh)
    bi = beaker_outer(fx, btop + 3, bw - 10, bh - 5)
    lv = BENCH - 4 - 44 * flow
    liquid_in(cv, bi, lv, c_fil)
    glass(cv, bo)
    # phễu
    fun = [(fx - 66, 90), (fx + 66, 90), (fx + 8, 176), (fx + 6, 224), (fx - 6, 224), (fx - 8, 176)]
    glass(cv, fun, hl=False)
    paper = [(fx - 58, 98), (fx + 58, 98), (fx + 4, 170)]
    cv.poly(paper, fill=(250, 250, 245, 235), outline=(190, 190, 180, 255), w=1.5)
    # chất rắn lắng trên giấy
    pile = flow * 20
    if pile > 0.5:
        cv.clip(paper, lambda c: c.ell(fx, 164, 22 + pile * 0.9, pile, fill=c_res))
    # dung dịch trong phễu
    lq = 118 + 36 * flow
    if t > P[1]:
        liquid_in(cv, [(fx - 56, 100), (fx + 56, 100), (fx + 6, 166)], lq - (0 if pour < 1 else 4 * (1 - flow)), (c_mix[0], c_mix[1], c_mix[2], 140))
    # giọt dịch lọc
    if flow > 0.04 and t < P[3] + 0.1 * T:
        drop_fall(cv, fx, 226, lv - 2, t, 0.5, c_fil if fold(sc["dich_loc"]["mau"]) not in ("khong mau",) else (160, 200, 230, 200), t0=P[1] + 0.05 * T)
    # cốc rót hỗn hợp
    pr = seg(t, P[0], P[1] * 0.9 + 0.001) if t < P[1] else 1.0
    tilt = -34 * seg(t, P[1] - 0.04 * T, P[1] + 0.08 * T) * (1 - seg(t, P[2] + 0.14 * T, P[2] + 0.22 * T))
    ox, oy = 470, 70
    body = [(ox - 42, oy), (ox - 36, oy + 76), (ox + 36, oy + 76), (ox + 42, oy)]
    inb = [(ox - 38, oy + 4), (ox - 33, oy + 73), (ox + 33, oy + 73), (ox + 38, oy + 4)]
    piv = (ox - 42, oy)
    body_r, inb_r = rot(body, tilt, piv), rot(inb, tilt, piv)
    cm = (c_mix[0], c_mix[1], c_mix[2], 200)
    full = 1 - 0.7 * seg(t, P[1], P[2] + 0.18 * T)
    cv.clip(inb_r, lambda c: c.poly(rot([(ox - 40, oy + 76 - 70 * full), (ox + 40, oy + 76 - 70 * full), (ox + 40, oy + 80), (ox - 40, oy + 80)], tilt, piv), fill=cm))
    rnd = random.Random(3)
    for i in range(16):
        cv.clip(inb_r, lambda c, i=i: c.ell(*rot([(ox + rnd.uniform(-28, 28), oy + 76 - rnd.uniform(4, 60) * full)], tilt, piv)[0], 2.4, 2.4, fill=(c_res[0], c_res[1], c_res[2], 220)))
    glass(cv, body_r, hl=False)
    if pour > 0.3:
        sx, sy = rot([(ox - 42, oy)], tilt, piv)[0]
        pts = [(sx, sy), (sx - 14, sy + 24), (fx + 8, 108), (fx + 4, 116)]
        cv.line(pts, (cm[0], cm[1], cm[2], 210), 4)
    tag(cv, 28, 62, f"Hỗn hợp: {mix['ten']}", (30, 90, 140))
    tag(cv, 28, 92, f"Chất rắn giữ lại: {mix['ten_ran']}", (170, 80, 30))
    tag(cv, 28, 122, f"Dịch lọc: {sc['dich_loc']['ten']}", (14, 110, 90))


# ================================================================ MÔ HÌNH 5: chuẩn độ
def draw_chuan_do(cv, t, sc, T):
    P = phases(T)
    titr = sc["chat_chuan"]
    ana = sc["chat_phan_tich"]
    c_t = liq(titr["mau"])
    c_before = liq(sc["mau_truoc"])
    c_after = liq(sc["mau_sau"])
    vend = sc["the_tich_cuoi"]
    run = seg(t, P[1], P[3] - 0.04 * T)           # tiến trình nhỏ giọt
    vol = vend * run
    bx = 330
    stand(cv, 220, 40, arm_y=110, arm_to=(bx - 14, 110))
    cv.line([(220, 110), (bx - 14, 110)], (98, 108, 120, 255), 5)
    # burette
    bt, bb = 40, 196
    bur = [(bx - 12, bt), (bx - 12, bb), (bx + 12, bb), (bx + 12, bt)]
    lvl = bt + 22 + (bb - bt - 40) * (vol / 50.0) * 1.8 if vend < 28 else bt + 22 + (bb - bt - 40) * (vol / (vend * 1.8))
    lvl = min(lvl, bb - 18)
    liquid_in(cv, bur, lvl, c_t)
    glass(cv, bur)
    for k in range(0, 11):
        yy = bt + 22 + k * (bb - bt - 44) / 10
        cv.line([(bx + 12, yy), (bx + (7 if k % 5 else 3), yy)], (90, 100, 112, 255), 1.5)
    cv.poly([(bx - 12, bb), (bx + 12, bb), (bx + 3, bb + 26), (bx - 3, bb + 26)], fill=(235, 245, 250, 200), outline=GLASS_LINE, w=1.5)
    cv.rrect(bx - 22, bb + 4, bx + 2, bb + 10, 2, fill=(70, 130, 220, 255))     # khóa burette
    tag(cv, bx + 24, 70, f"V = {vol:.1f} mL", (40, 100, 150))
    # bình tam giác trên nền trắng
    cv.rrect(bx - 80, BENCH - 8, bx + 80, BENCH + 4, 2, fill=(250, 250, 250, 255), outline=(200, 200, 200, 255), w=1)
    fo = flask_outer(bx, BENCH - 106, 36, 40, 124, 106)
    fi = flask_outer(bx, BENCH - 104, 30, 38, 118, 102)
    k_c = seg(t, P[3] - 0.30 * T, P[3] - 0.12 * T)
    col = lerp(c_before, c_after, k_c)
    swirl = 6 * math.sin(t * 5)
    flv = BENCH - 56 - 0.18 * vol * 0.4
    liquid_in(cv, fi, flv, col)
    if run < 0.995 and t > P[1]:
        drop_fall(cv, bx, bb + 28, flv - 2, t, 0.45, c_t if fold(titr["mau"]) not in ("khong mau",) else (170, 205, 235, 200), t0=P[1], t1=P[3] - 0.04 * T)
        if k_c < 0.5 and run > 0.7:
            cv.clip(fi, lambda c: c.ell(bx + swirl, flv + 6, 16, 5, fill=(c_after[0], c_after[1], c_after[2], 90)))
    glass(cv, fo)
    tag(cv, 28, 62, f"Burette: {titr['ten']}", (30, 90, 140))
    tag(cv, 28, 92, f"Bình tam giác: {ana['ten']}", (170, 80, 30))
    if sc.get("chi_thi"):
        tag(cv, 28, 122, f"Chất chỉ thị: {sc['chi_thi']}", (14, 110, 90))
    if k_c > 0.5:
        tag(cv, bx + 90, 200, "Đổi màu: điểm tương đương", (190, 40, 40))


# ================================================================ MÔ HÌNH 6: đốt cháy
def draw_chay(cv, t, sc, T):
    P = phases(T)
    solid = liq(sc["chat_chay"]["mau"])
    solid = (solid[0], solid[1], solid[2], 255)
    prod = sc.get("san_pham") or {"ten": "", "mau": "trắng"}
    pcol = liq(prod["mau"])
    pcol = (pcol[0], pcol[1], pcol[2], 255)
    fcol = flame_rgb(sc["mau_lua"])
    oxy = sc["moi_truong"] == "oxi"
    jx = 560
    # lọ thu khí
    jar = [(jx - 62, 120), (jx - 62, BENCH), (jx + 62, BENCH), (jx + 62, 120)]
    if oxy:
        cv.clip(jar, lambda c: c.rrect(jx - 64, 120, jx + 64, BENCH + 2, 0, fill=(190, 220, 245, 70)))
    glass(cv, jar)
    cv.rrect(jx - 70, 112, jx + 70, 122, 3, fill=(230, 242, 248, 210), outline=GLASS_LINE, w=1.5)
    # đèn cồn
    lamp_x = 150
    lamp(cv, lamp_x, on=seg(t, P[0] + 0.5, P[0] + 1.5), t=t)
    # đường đi của muỗng đốt
    heat_p = seg(t, P[0] + 0.8, P[1])                 # đưa vào ngọn lửa đèn
    move = seg(t, P[1] + 0.2 * (P[2] - P[1]), P[2] + 0.02 * T)      # chuyển vào lọ
    pos_a = (lamp_x + 4, 258)
    pos_b = (jx, 224)
    px = pos_a[0] + (pos_b[0] - pos_a[0]) * move
    py = pos_a[1] + (pos_b[1] - pos_a[1]) * move - math.sin(move * math.pi) * 40
    py -= 30 * (1 - heat_p)
    ign = seg(t, P[1], P[1] + 0.16 * (P[2] - P[1]))
    burn = seg(t, P[1] + 0.12 * (P[2] - P[1]), P[2] + 0.1 * T) * (1 - seg(t, P[3] - 0.02 * T, P[3] + 0.12 * T) * 0.85)
    in_jar = move > 0.85
    cv.line([(px - 210, py - 70 + 40 * 0), (px, py)], (95, 105, 118, 255), 4)
    cv.rrect(px - 18, py, px + 18, py + 7, 3, fill=(150, 156, 165, 255))
    after = seg(t, P[3] - 0.02 * T, P[3] + 0.1 * T)
    mcol = lerp(solid, pcol, after)
    cv.ell(px, py - 4, 9, 6, fill=mcol, outline=(60, 60, 60, 150), w=1)
    # lửa
    if ign > 0.05 and burn > 0.03:
        size = (34 + (46 if in_jar else 0)) * burn * (0.6 + 0.4 * ign)
        flame(cv, px, py - 6, size, t, fcol, glow=True, w=size * 0.32)
        if sc["tia_lua"] and burn > 0.3:
            rnd = random.Random(9)
            for i in range(26):
                a = rnd.uniform(0, 2 * math.pi)
                sp = rnd.uniform(35, 105)
                u = ((t * 1.4 + rnd.random()) % 1.0)
                sx = px + math.cos(a) * sp * u
                sy = py - 8 + math.sin(a) * sp * u * 0.8 + 70 * u * u
                cv.ell(sx, sy, 1.9, 1.9, fill=(255, 220, 130, int(255 * (1 - u))))
    # khói
    smoke_kind = sc["khoi"]
    if smoke_kind != "khong" and burn > 0.15:
        col = (240, 240, 240) if smoke_kind == "trang" else (150, 110, 70)
        if in_jar:
            cv.clip(jar, lambda c: smoke(c, jx, py + 10, t, n=14, color=col, seed=5, spread=40, rise=110, alpha=int(135 * burn)))
            fog = seg(t, P[2], P[3])
            cv.clip(jar, lambda c: c.rrect(jx - 64, 120, jx + 64, BENCH + 2, 0, fill=(col[0], col[1], col[2], int(70 * fog))))
        else:
            smoke(cv, px, py - 20, t, n=8, color=col, seed=5, spread=20, rise=90, alpha=int(110 * burn))
    tag(cv, 28, 62, f"Đốt: {sc['chat_chay']['ten']}", (30, 90, 140))
    tag(cv, 28, 92, "Trong bình khí oxi" if oxy else "Trong không khí", (170, 80, 30))
    if prod.get("ten"):
        tag(cv, 28, 122, f"Sản phẩm: {prod['ten']}", (14, 110, 90))


DRAWERS = {"ong_nghiem": draw_ong_nghiem, "khi": draw_khi, "chung_cat": draw_chung_cat, "loc": draw_loc,
           "chuan_do": draw_chuan_do, "chay": draw_chay}


# ================================================================ chuẩn hóa tham số do AI trả về
def _s(v, n=80, default=""):
    v = str(v if v is not None else default).strip()
    return v[:n]


def _b(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "có", "co", "yes", "1")
    return bool(v) if v is not None else default


def _chat(d, ten_default="", mau_default="không màu"):
    d = d if isinstance(d, dict) else {}
    return {"ten": _s(d.get("ten"), 40, ten_default), "mau": color_ok(d.get("mau"), mau_default)}


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


BLOCK = ["thuoc no", "chat no", "bom", "ma tuy", "heroin", "methamphetamine", "sarin", "vx ", "khi doc chien", "chat doc than kinh", "napalm",
         "che tao vu khi", "tu che", "tnt", "c4", "nitroglycerin", "thuoc sung", "tu sat", "dau doc"]


def norm_scene(raw):
    """Kiểm tra tham số AI trả về. Trả (scene, None) hoặc (None, lý do không mô phỏng được)."""
    raw = raw if isinstance(raw, dict) else {}
    blob = fold(" ".join(str(v) for v in raw.values() if isinstance(v, (str, int, float))))
    if any(b in blob for b in BLOCK):
        return None, "Thí nghiệm này không phù hợp để mô phỏng trong ứng dụng học tập."
    m = fold(raw.get("mo_hinh"))
    if m not in MODELS:
        return None, _s(raw.get("ly_do"), 200) or ("Chưa có mô hình mô phỏng phù hợp. Hiện hỗ trợ: phản ứng trong ống nghiệm hoặc cốc, "
                                                  "điều chế và thu khí, chưng cất, lọc, chuẩn độ, đốt cháy.")
    sc = {"mo_hinh": m, "tieu_de": _s(raw.get("tieu_de"), 80, "Thí nghiệm hóa học")}
    buoc = [_s(x, 110) for x in (raw.get("buoc") or []) if _s(x)][:4]
    defaults = ["Chuẩn bị dụng cụ và hóa chất.", "Tiến hành thí nghiệm.", "Quan sát phản ứng xảy ra.", "Ghi lại hiện tượng."]
    sc["buoc"] = (buoc + defaults[len(buoc):])[:4]
    sc["hien_tuong"] = [_s(x, 130) for x in (raw.get("hien_tuong") or []) if _s(x)][:5] or ["Quan sát và ghi lại hiện tượng."]
    pt = raw.get("phuong_trinh") if isinstance(raw.get("phuong_trinh"), dict) else {}
    side = lambda L: [{"he_so": max(1, int(_num(x.get("he_so"), 1, 99, 1))), "smiles": _s(x.get("smiles"), 80)} for x in (L or []) if isinstance(x, dict)][:6]
    sc["phuong_trinh"] = {"text": _s(pt.get("text"), 160), "trai": side(pt.get("trai")), "phai": side(pt.get("phai"))}
    sc["an_toan"] = [_s(x, 130) for x in (raw.get("an_toan") or []) if _s(x)][:3]
    if m == "ong_nghiem":
        th = raw.get("them") if isinstance(raw.get("them"), dict) else {}
        sc.update({"vat": "coc" if fold(raw.get("vat")) == "coc" else "ong_nghiem", "chat_dau": _chat(raw.get("chat_dau"), "dung dịch")})
        sc["them"] = {"ten": _s(th.get("ten"), 40), "kieu": "ran" if fold(th.get("kieu")) == "ran" else "long", "mau": color_ok(th.get("mau"), "không màu")}
        sc["mau_sau"] = color_ok(raw.get("mau_sau"), "") or None if raw.get("mau_sau") else None
        sc["khi"] = _b(raw.get("khi"))
        sc["khi_mau"] = color_ok(raw.get("khi_mau"), "không màu")
        sc["ket_tua"] = color_ok(raw.get("ket_tua"), "trắng") if raw.get("ket_tua") and fold(raw.get("ket_tua")) not in ("none", "khong", "null") else None
        sc["ran_tan"] = _b(raw.get("ran_tan"))
        sc["bam"] = color_ok(raw.get("bam"), "đỏ nâu") if raw.get("bam") and fold(raw.get("bam")) not in ("none", "khong", "null") else None
        sc["dun_nong"] = _b(raw.get("dun_nong"))
        sc["bay"] = _b(raw.get("bay"))
    elif m == "khi":
        cr = raw.get("chat_ran") if isinstance(raw.get("chat_ran"), dict) else {}
        sc["cach"] = "ran_long" if fold(raw.get("cach")) == "ran_long" else "dun_ran"
        sc["chat_ran"] = {"ten": _s(cr.get("ten"), 40, "chất rắn"), "mau": color_ok(cr.get("mau"), "trắng"),
                          "mau_sau": color_ok(cr.get("mau_sau"), "") if cr.get("mau_sau") else None}
        sc["chat_long"] = _chat(raw.get("chat_long"), "dung dịch")
        sc["khi"] = _chat(raw.get("khi"), "khí")
        thu = fold(raw.get("thu"))
        sc["thu"] = thu if thu in ("day_nuoc", "day_kk_up", "day_kk_ngua") else "day_nuoc"
        tt = fold(raw.get("thu_thu"))
        sc["thu_thu"] = tt if tt in ("que_dom", "tieng_no_nho") else "none"
    elif m == "chung_cat":
        sc["chat_long"] = _chat(raw.get("chat_long"), "hỗn hợp lỏng")
        sc["chat_cat"] = _chat(raw.get("chat_cat"), "chất lỏng cất được")
        sc["nhiet_do_soi"] = _num(raw.get("nhiet_do_soi"), 35, 250, 78)
    elif m == "loc":
        h = raw.get("hon_hop") if isinstance(raw.get("hon_hop"), dict) else {}
        sc["hon_hop"] = {"ten": _s(h.get("ten"), 40, "hỗn hợp"), "ten_ran": _s(h.get("ten_ran"), 40, "chất rắn"),
                         "mau_dd": color_ok(h.get("mau_dd"), "đục trắng"), "mau_ran": color_ok(h.get("mau_ran"), "nâu")}
        sc["dich_loc"] = _chat(raw.get("dich_loc"), "dịch lọc")
    elif m == "chuan_do":
        sc["chat_chuan"] = _chat(raw.get("chat_chuan"), "dung dịch chuẩn")
        sc["chat_phan_tich"] = _chat(raw.get("chat_phan_tich"), "dung dịch cần chuẩn độ")
        sc["chi_thi"] = _s(raw.get("chi_thi"), 40)
        sc["mau_truoc"] = color_ok(raw.get("mau_truoc"), "không màu")
        sc["mau_sau"] = color_ok(raw.get("mau_sau"), "hồng")
        sc["the_tich_cuoi"] = _num(raw.get("the_tich_cuoi"), 5, 45, 20)
    elif m == "chay":
        sc["chat_chay"] = _chat(raw.get("chat_chay"), "chất rắn", "xám")
        sc["moi_truong"] = "oxi" if fold(raw.get("moi_truong")) == "oxi" else "khong_khi"
        sc["mau_lua"] = _s(raw.get("mau_lua"), 20, "vàng")
        kh = fold(raw.get("khoi"))
        sc["khoi"] = kh if kh in ("trang", "nau") else "khong"
        sc["tia_lua"] = _b(raw.get("tia_lua"))
        sc["san_pham"] = _chat(raw.get("san_pham"), "", "trắng")
    return sc, None


def describe(sc):
    """Mô tả những gì video sẽ thể hiện, lấy trực tiếp từ tham số (không phải lời AI)."""
    m = sc["mo_hinh"]
    out = []
    if m == "ong_nghiem":
        out.append(f"{'Cốc' if sc['vat'] == 'coc' else 'Ống nghiệm'} chứa {sc['chat_dau']['ten']} ({sc['chat_dau']['mau']})")
        if sc["them"]["ten"]:
            out.append(f"Thêm {sc['them']['ten']} ({'chất rắn' if sc['them']['kieu'] == 'ran' else 'dung dịch nhỏ giọt'}, màu {sc['them']['mau']})")
        if sc["khi"]:
            out.append("Có bọt khí thoát ra" + (f" ({sc['khi_mau']})" if fold(sc["khi_mau"]) not in ("khong mau", "trong suot") else ""))
        if sc["ket_tua"]:
            out.append(f"Xuất hiện kết tủa màu {sc['ket_tua']}")
        if sc.get("mau_sau"):
            out.append(f"Dung dịch đổi từ {sc['chat_dau']['mau']} sang {sc['mau_sau']}")
        if sc["ran_tan"]:
            out.append("Chất rắn tan dần")
        if sc["bam"]:
            out.append(f"Chất rắn bị phủ lớp màu {sc['bam']}")
        if sc["bay"]:
            out.append("Chất rắn chạy trên mặt dung dịch")
        if sc["dun_nong"]:
            out.append("Đun nóng bằng đèn cồn")
    elif m == "khi":
        out.append("Điều chế " + sc["khi"]["ten"] + (" bằng cách đun nóng " + sc["chat_ran"]["ten"] if sc["cach"] == "dun_ran" else f" từ {sc['chat_ran']['ten']} và {sc['chat_long']['ten']}"))
        out.append({"day_nuoc": "Thu khí bằng cách đẩy nước", "day_kk_up": "Thu khí bằng cách đẩy không khí (bình để ngửa)", "day_kk_ngua": "Thu khí bằng cách đẩy không khí (bình úp)"}[sc["thu"]])
        if sc["thu_thu"] == "que_dom":
            out.append("Thử khí bằng que đóm còn tàn đỏ")
        elif sc["thu_thu"] == "tieng_no_nho":
            out.append("Thử khí bằng ngọn lửa (cháy kèm tiếng nổ nhỏ)")
    elif m == "chung_cat":
        out.append(f"Đun hỗn hợp {sc['chat_long']['ten']}, nhiệt độ tăng tới khoảng {sc['nhiet_do_soi']:.0f} °C rồi sôi")
        out.append(f"Hơi ngưng tụ trong ống sinh hàn, thu {sc['chat_cat']['ten']} ở bình hứng")
    elif m == "loc":
        out.append(f"Rót hỗn hợp {sc['hon_hop']['ten']} qua phễu có giấy lọc")
        out.append(f"{sc['hon_hop']['ten_ran']} ở lại trên giấy lọc, dịch lọc ({sc['dich_loc']['ten']}) chảy xuống cốc")
    elif m == "chuan_do":
        out.append(f"Nhỏ {sc['chat_chuan']['ten']} từ burette vào {sc['chat_phan_tich']['ten']}")
        out.append(f"Màu đổi từ {sc['mau_truoc']} sang {sc['mau_sau']} ở điểm tương đương (V ≈ {sc['the_tich_cuoi']:.0f} mL)")
    elif m == "chay":
        out.append(f"Đốt {sc['chat_chay']['ten']} {'trong bình khí oxi' if sc['moi_truong'] == 'oxi' else 'trong không khí'}, ngọn lửa màu {sc['mau_lua']}")
        if sc["tia_lua"]:
            out.append("Có tia lửa bắn ra")
        if sc["khoi"] != "khong":
            out.append("Có khói " + ("trắng" if sc["khoi"] == "trang" else "nâu"))
    return out


# ================================================================ dựng khung hình và video
def duration(sc):
    return {"ong_nghiem": 14.0, "khi": 17.0, "chung_cat": 16.0, "loc": 14.0, "chuan_do": 15.0, "chay": 15.0}[sc["mo_hinh"]]


def render_frame(sc, t, T=None):
    T = T or duration(sc)
    cv = Cv(background())
    P = phases(T)
    DRAWERS[sc["mo_hinh"]](cv, min(t, T), sc, T)
    idx = 0 if t < P[1] else 1 if t < P[2] else 2 if t < P[3] else 3
    title_bar(cv, sc["tieu_de"])
    caption_bar(cv, sc["buoc"][idx], idx, 4)
    result_card(cv, sc, seg(t, T + 0.0, T + 0.0001) if False else 0)
    return cv.im


def render_frames(sc, fps=FPS, progress=None, hold=3.5):
    """Sinh lần lượt các khung hình (PIL.Image đã thu nhỏ). hold = số giây hiện thẻ kết quả ở cuối."""
    T = duration(sc)
    total = T + hold
    n = int(total * fps)
    for i in range(n):
        t = i / fps
        cv = Cv(background())
        DRAWERS[sc["mo_hinh"]](cv, min(t, T), sc, T)
        P = phases(T)
        idx = 0 if t < P[1] else 1 if t < P[2] else 2 if t < P[3] else 3
        title_bar(cv, sc["tieu_de"])
        caption_bar(cv, sc["buoc"][idx], idx, 4)
        result_card(cv, sc, seg(t, T - 0.5, T + 0.6))
        if progress:
            progress((i + 1) / n)
        yield cv.im.resize((W, H), Image.LANCZOS)


def render_mp4(sc, fps=FPS, progress=None):
    import imageio
    import numpy as np
    path = tempfile.mktemp(suffix=".mp4")
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16,
                                ffmpeg_params=["-movflags", "+faststart"]) as w:
            for im in render_frames(sc, fps, progress):
                w.append_data(np.asarray(im))
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def render_gif(sc, fps=10, progress=None):
    frames = [im.resize((640, 358), Image.LANCZOS).quantize(colors=96, method=Image.MEDIANCUT) for im in render_frames(sc, fps, progress)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return buf.getvalue()


def render_video(sc, progress=None):
    """Trả về (bytes, 'video/mp4' hoặc 'image/gif')."""
    try:
        return render_mp4(sc, progress=progress), "video/mp4"
    except Exception:
        return render_gif(sc, progress=progress), "image/gif"


# ================================================================ lời dặn cho AI
SCENE_PROMPT = """Bạn là bộ PHÂN TÍCH THÍ NGHIỆM HÓA HỌC phổ thông. Nhiệm vụ: đọc mô tả thí nghiệm của học sinh và điền THAM SỐ để máy vẽ video mô phỏng. Bạn không viết mã, không vẽ.
Chọn đúng MỘT mô hình trong: ong_nghiem, khi, chung_cat, loc, chuan_do, chay. Nếu thí nghiệm không khớp mô hình nào, hoặc nguy hiểm (chất nổ, chất độc, ma túy, vũ khí), trả {"mo_hinh":"khong_ho_tro","ly_do":"giải thích ngắn bằng tiếng Việt"}.
Màu chỉ dùng các từ: không màu, trắng, đục trắng, xanh lam, xanh lam nhạt, xanh lục, vàng, vàng nhạt, cam, đỏ, nâu đỏ, nâu, tím, tím đậm, hồng, đen, xám, bạc, xanh đen.
Màu lửa chỉ dùng: trắng chói, vàng, cam, đỏ cam, xanh lam nhạt, tím nhạt, xanh lục.
Trường chung (mọi mô hình): "tieu_de" (ngắn), "buoc" (đúng 4 câu thuyết minh cho 4 giai đoạn: chuẩn bị, thêm chất hoặc bắt đầu, phản ứng diễn ra, quan sát), "hien_tuong" (2-5 câu hiện tượng đúng với SGK), "phuong_trinh": {"text":"phương trình đã cân bằng, viết A + B → C","trai":[{"he_so":2,"smiles":"SMILES"}],"phai":[...]}, "an_toan" (1-3 lưu ý an toàn).
Mô hình ong_nghiem (phản ứng trong ống nghiệm hoặc cốc): "vat":"ong_nghiem"|"coc", "chat_dau":{"ten","mau"} (chất có sẵn trong vật chứa), "them":{"ten","kieu":"ran"|"long","mau"}, "mau_sau" (màu dung dịch lúc cuối, bỏ trống nếu không đổi), "khi":true/false, "khi_mau", "ket_tua" (màu kết tủa hoặc null), "ran_tan":true/false, "bam" (màu lớp phủ lên chất rắn hoặc null), "dun_nong":true/false, "bay":true/false (kim loại kiềm chạy trên mặt nước, dùng vat "coc").
Mô hình khi (điều chế và thu khí): "cach":"dun_ran"|"ran_long", "chat_ran":{"ten","mau","mau_sau"}, "chat_long":{"ten","mau"} (chỉ khi ran_long), "khi":{"ten","mau"}, "thu":"day_nuoc"|"day_kk_up"|"day_kk_ngua", "thu_thu":"que_dom"|"tieng_no_nho"|"none".
Mô hình chung_cat: "chat_long":{"ten","mau"}, "chat_cat":{"ten","mau"}, "nhiet_do_soi": số °C.
Mô hình loc: "hon_hop":{"ten","ten_ran","mau_dd","mau_ran"}, "dich_loc":{"ten","mau"}.
Mô hình chuan_do: "chat_chuan":{"ten","mau"}, "chat_phan_tich":{"ten","mau"}, "chi_thi", "mau_truoc", "mau_sau", "the_tich_cuoi": số mL.
Mô hình chay: "chat_chay":{"ten","mau"}, "moi_truong":"khong_khi"|"oxi", "mau_lua", "khoi":"trang"|"nau"|"khong", "tia_lua":true/false, "san_pham":{"ten","mau"}.
Chỉ trả về MỘT đối tượng JSON, không markdown."""
