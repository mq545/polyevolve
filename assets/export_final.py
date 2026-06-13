"""Export the final PolyEvolve (Evo-graph) assets: mark sizes + light/dark wordmarks."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

S = 4
TILE = 512
ARIALBD = "/mnt/c/Windows/Fonts/arialbd.ttf"
INDIGO, VIOLET, CYAN, WHITE = (99, 102, 241), (139, 92, 246), (34, 211, 238), (255, 255, 255)
INK = (17, 24, 39)
VIOLET_LT = (167, 139, 250)  # readable violet on dark


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(w, h):
    g = Image.new("RGB", (128, 128))
    px = g.load()
    for y in range(128):
        for x in range(128):
            t = (x + y) / 254.0
            px[x, y] = lerp(INDIGO, VIOLET, t * 2) if t < 0.5 else lerp(VIOLET, CYAN, (t - 0.5) * 2)
    return g.resize((w, h), Image.BILINEAR)


def mark():
    w = TILE * S
    out = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    m = Image.new("L", (w, w), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, w - 1], radius=int(w * 0.22), fill=255)
    out.paste(gradient(w, w), (0, 0), m)
    d = ImageDraw.Draw(out)
    P = lambda fx, fy: (int(fx * w), int(fy * w))  # noqa: E731
    g0, g1a, g1b = P(0.30, 0.50), P(0.52, 0.34), P(0.52, 0.66)
    g2a, g2b, g2c = P(0.74, 0.26), P(0.74, 0.50), P(0.74, 0.74)
    lw = int(w * 0.0125)
    for a, b in [(g0, g1a), (g0, g1b), (g1a, g2a), (g1a, g2b), (g1b, g2b), (g1b, g2c)]:
        d.line([a, b], fill=(255, 255, 255, 170), width=lw)
    r = int(w * 0.045)
    for p in (g0, g1a, g1b, g2a, g2c):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=WHITE)
    cr = int(w * 0.10)
    d.ellipse(
        [g2b[0] - cr, g2b[1] - cr, g2b[0] + cr, g2b[1] + cr], outline=(255, 255, 255, 128), width=lw
    )
    br = int(r * 1.7)
    d.ellipse([g2b[0] - br, g2b[1] - br, g2b[0] + br, g2b[1] + br], fill=WHITE)
    return out.resize((TILE, TILE), Image.LANCZOS)


def wordmark(mk, poly_col, evolve_col, transparent=True):
    H = 220
    pad = 28
    msz = H - pad * 2
    sm = mk.resize((msz, msz), Image.LANCZOS)
    fnt = ImageFont.truetype(ARIALBD, 104)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    w1 = tmp.textlength("Poly", font=fnt)
    w2 = tmp.textlength("Evolve", font=fnt)
    W = pad + msz + 34 + int(w1 + w2) + pad
    bg = (0, 0, 0, 0) if transparent else (255, 255, 255, 255)
    img = Image.new("RGBA", (W, H), bg)
    img.alpha_composite(sm, (pad, pad))
    d = ImageDraw.Draw(img)
    tx, ty = pad + msz + 34, (H - 104) // 2 - 8
    d.text((tx, ty), "Poly", font=fnt, fill=poly_col)
    d.text((tx + w1, ty), "Evolve", font=fnt, fill=evolve_col)
    return img


mk = mark()
for sz in (1024, 512, 256, 128):
    mk.resize((sz, sz), Image.LANCZOS).save(f"assets/logo-mark-{sz}.png")
mk.resize((512, 512), Image.LANCZOS).save("assets/logo-mark.png")
wordmark(mk, INK, VIOLET).save("assets/logo-wordmark.png")  # light theme
wordmark(mk, WHITE, VIOLET_LT).save("assets/logo-wordmark-dark.png")  # dark theme
print("exported final Evo-graph assets")
