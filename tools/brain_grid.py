"""One think as a grid of brain pictures, 2 across and 6 down, for the README.

  uvicorn app.server:app --port 8000                 # in WSL first, then on Windows:
  python tools/brain_grid.py                         # forward: the backprop fly answers 1.e4
  python tools/brain_grid.py --backward runs/bptt/e4.npz   # the error going back (tools/bptt.py writes it)

It opens the app in headless Chrome, draws the brain with the app's own BrainView at every
step and screenshots the grid with a caption on every picture. The captions come from the
measured activity per region, the phrases are mine.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# What happens at each step of the backprop fly's answer to 1.e4, read off which regions grew most
# since the step before (the numbers under each caption). Photoreceptors sit in the eye, and only 28
# of the 4,020 have a known cell body, so step 1 has nothing to show in this picture.
FORWARD = ["only the photoreceptors, which sit in the eye, outside this picture", "the optic lobes start to answer",
           "the optic lobes and the visual projection neurons pick it up", "visual projection neurons carry it into the brain",
           "the reward dopamine neurons join in", "the mushroom body and the central complex light up",
           "the Kenyon cells jump the most", "the antennal lobe and the punishment dopamine neurons jump",
           "the board is everywhere now, and still growing", "the antennal lobe, where a fly smells, keeps catching up",
           "the antennal lobe and the central complex keep climbing",
           "the Kenyon cells jump again, and the readout reads the descending neurons"]
DOTS = ('<span><i style="background:#f5614f"></i>firing more than usual for this board</span>'
        '<span><i style="background:#4d99ff"></i>less</span><span><i style="background:#575c66"></i>as usual</span>')
LEGEND_FORWARD = (DOTS + '<br>Under each caption: the regions that grew most since the step before. A region\'s number is how '
                  'far its neurons are from their usual rate, on average, and a rate goes from 0 to 1.')

PAGE = """
async ({ bytes, span, w, h, order }) => {
  const { BrainView } = await import('/js/brain.js')
  const meta = await (await fetch('/api/brain')).json()
  const points = await (await fetch('/api/brain/points')).arrayBuffer()
  const canvas = Object.assign(document.createElement('canvas'), { width: w, height: h })
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:${h}px;z-index:99`
  document.body.append(canvas)
  const view = new BrainView(canvas, meta, points)
  view.spin = false
  view.resize()
  view.setActivity(Uint8Array.from(atob(bytes), (c) => c.charCodeAt(0)))
  if (span) view.material.uniforms.uSpan.value = span
  const urls = []
  for (const s of order) {
    view.setStep(s)
    view.renderer.render(view.scene, view.camera)
    urls.push(canvas.toDataURL('image/png'))
  }
  return urls
}
"""

GRID = """
({ urls, captions, legend, w, h }) => {
  document.body.replaceChildren()
  document.body.style.cssText = 'margin:0;background:#302e2b;display:block'
  const grid = document.createElement('div')
  grid.id = 'grid'
  grid.style.cssText = `display:grid;grid-template-columns:repeat(2,${w}px);gap:12px;padding:16px;background:#262522;width:max-content;font-family:"Segoe UI",system-ui,sans-serif`
  urls.forEach((url, i) => {
    const fig = document.createElement('figure')
    fig.style.cssText = 'margin:0;border-radius:8px;overflow:hidden;background:#1b1a18'
    const cap = document.createElement('figcaption')
    cap.style.cssText = 'padding:12px 16px 4px;color:#fff;font-size:20px;line-height:1.3'
    cap.innerHTML = `<b>${captions[i].title}</b>${captions[i].phrase ? ': ' + captions[i].phrase : ''}`
      + `<div style="color:#989795;font-size:15px;margin-top:3px">${captions[i].numbers}</div>`
    const img = Object.assign(document.createElement('img'), { src: url, width: w, height: h })
    img.style.display = 'block'
    fig.append(cap, img)
    grid.append(fig)
  })
  const foot = document.createElement('p')
  foot.style.cssText = 'grid-column:1/3;margin:2px 4px 0;color:#c3c2c1;font-size:17px;line-height:1.5'
  foot.innerHTML = legend
  for (const s of foot.querySelectorAll('span')) s.style.cssText = 'margin-right:18px;white-space:nowrap'
  for (const i of foot.querySelectorAll('i')) i.style.cssText += ';display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px'
  grid.append(foot)
  document.body.append(grid)
}
"""


def grew(now: dict, before: dict, top: int = 3) -> str:
    """The regions that grew most since the step before ('other' is a leftover bin, so it's skipped)."""
    up = sorted(((now[r] - before.get(r, 0.0), r) for r in now if r != "other"), reverse=True)
    up = [(v, r) for v, r in up[:top] if v >= 0.0005]
    text = ", ".join(f"{r} +{v:.3f}" for v, r in up)
    return text + (", nothing else yet" if len(up) == 1 else "")


def forward(page) -> tuple[str, list, list[dict]]:
    """Play 1.e4 against the backprop fly; its brain's bytes and the captions."""
    res = page.evaluate("""async (fen) => {
      const r = await fetch('/api/play', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen, uci: 'e2e4', fly: 'dl', seen: [] }) })
      const play = await r.json()
      const buf = new Uint8Array(await (await fetch(`/api/activity/${play.fly.trace}`)).arrayBuffer())
      let s = ''
      for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000))
      return { bytes: btoa(s), activity: play.fly.activity, move: play.fly.move.san }
    }""", START)
    act = res["activity"]
    caps = [{"title": f"Step {s + 1} of 12", "phrase": FORWARD[s], "numbers": grew(act[s], act[s - 1] if s else {})}
            for s in range(12)]
    print(f"the fly answered {res['move']}")
    for c in caps:
        print(f"{c['title']}: {c['phrase']} | {c['numbers']}")
    return res["bytes"], list(range(12)), caps


def backward(path: Path) -> tuple[str, list, list[dict], float]:
    """The error's way back (tools/bptt.py): steps 12 down to 1."""
    d = np.load(path)
    caps = json.loads(str(d["captions"]))
    return base64.b64encode(d["bytes"].tobytes()).decode(), list(range(11, -1, -1)), caps, float(d["span"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backward", type=Path, help="an .npz from tools/bptt.py: draw the error going back instead")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--size", type=int, nargs=2, default=(460, 500), help="one picture, CSS pixels (drawn at 1.5x)")
    args = ap.parse_args()
    w, h = args.size
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--use-angle=d3d11"])   # without ANGLE on D3D11 WebGL draws nothing headless
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
        page.goto(URL)
        page.wait_for_function("() => window.flyChess !== undefined")
        if args.backward:
            bytes_, order, caps, span = backward(args.backward)
            legend = json.loads(str(np.load(args.backward)["legend"]))
            out = args.out or Path("docs/img/brain_backward_grid.png")
        else:
            bytes_, order, caps = forward(page)
            span, legend = None, LEGEND_FORWARD
            out = args.out or Path("docs/img/brain_steps_grid.png")
        urls = page.evaluate(PAGE, {"bytes": bytes_, "span": span, "w": w, "h": h, "order": order})
        page.evaluate(GRID, {"urls": urls, "captions": caps, "legend": legend, "w": w, "h": h})
        page.wait_for_timeout(300)
        page.locator("#grid").screenshot(path=str(out))
        browser.close()
    Image.open(out).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)  # 4.9 MB -> about 1.3
    print(f"saved {out}")


if __name__ == "__main__":
    main()
