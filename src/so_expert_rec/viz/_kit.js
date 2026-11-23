const NS = "http://www.w3.org/2000/svg";
const ROLE_NAME = { tag: "Tag", question: "Question", user: "User", sink: "Absorbing sink" };

/* ---------- DOM helpers (all text via textContent; payload labels are untrusted data) ---------- */
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function H(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const fmt = (v, dp) => v == null ? "–" : (dp != null ? v.toFixed(dp) : (Math.abs(v) >= 1000 ? v.toLocaleString() : (Math.abs(v) >= 0.01 || v === 0 ? +v.toFixed(3) + "" : v.toExponential(2))));
function niceTicks(max, n) {
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag;
  const ticks = []; for (let v = 0; v <= max + step * 0.999; v += step) { ticks.push(+v.toPrecision(10)); if (v >= max) break; }
  return ticks;
}

/* ---------- tooltip: value leads, label follows; same details on keyboard focus ---------- */
const tipEl = H("div", "tip", null, document.body); tipEl.hidden = true;
function showTip(title, rows) {
  tipEl.replaceChildren();
  if (title) H("div", "tip-title", title, tipEl);
  for (const r of rows) {
    const row = H("div", "tip-row", null, tipEl);
    if (r.color) { const k = H("span", "key", null, row); k.style.borderTopColor = r.color; if (r.dash) k.style.borderTopStyle = "dashed"; }
    H("span", "v", r.v, row); H("span", "k", r.k, row);
  }
  tipEl.hidden = false;
}
function moveTip(x, y) {
  const r = tipEl.getBoundingClientRect(), pad = 14;
  if (x + pad + r.width > innerWidth - 8) x = x - r.width - pad * 2;
  if (y + pad + r.height > innerHeight - 8) y = y - r.height - pad * 2;
  tipEl.style.left = Math.max(8, x + pad) + "px"; tipEl.style.top = Math.max(8, y + pad) + "px";
}
function bindTip(el, getter, onOn, onOff) {
  el.addEventListener("pointermove", ev => { const t = getter(ev); if (!t) return; showTip(t.title, t.rows); moveTip(ev.clientX, ev.clientY); onOn && onOn(ev); });
  el.addEventListener("pointerleave", () => { tipEl.hidden = true; onOff && onOff(); });
  el.addEventListener("focus", () => { const t = getter(null); if (!t) return; const b = el.getBoundingClientRect(); showTip(t.title, t.rows); moveTip(b.right, b.top); onOn && onOn(null); });
  el.addEventListener("blur", () => { tipEl.hidden = true; onOff && onOff(); });
}

