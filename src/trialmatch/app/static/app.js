"use strict";
/* TrialMatch screening prototype: vanilla JS, no dependencies, no network except the local API. */
const $ = (id) => document.getElementById(id);
const S = { status: null, sentences: [], note: "", crit: [], job: null, pollTimer: null, selSent: null, selCard: null,
            examples: { notes: [], trials: [] }, es: null, labJob: null };

/* ---------- tiny DOM helper (never innerHTML with data) ---------- */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const SVGNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  for (const c of kids.flat()) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const ICONS = {
  check: "M4 12l5 5 11-11", x: "M6 6l12 12M18 6L6 18", question: "M9 9a3 3 0 116 0c0 2-3 2-3 5M12 18v.5",
  slash: "M5 19L19 5", warning: "M12 4L3 20h18zM12 10v5M12 17.5v.5",
};
function icon(name) {
  return sv("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2.6, "stroke-linecap": "round",
                     "stroke-linejoin": "round", "aria-hidden": "true" }, sv("path", { d: ICONS[name] || ICONS.warning }));
}
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(d);

async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); } catch (e) { throw new Error("Cannot reach the app server. Is it still running?"); }
  let data = null;
  try { data = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) { const er = new Error((data && data.error && data.error.message) || `Request failed (${r.status}).`); er.details = data && data.error && data.error.details; throw er; }
  return data;
}
const post = (path, obj) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj || {}) });

function alertBox(msg, kind = "error", host = "alerts") {
  $(host).append(h("div", { class: `alert ${kind}`, role: kind === "error" ? "alert" : "status" }, msg));
}
function clearAlerts(host = "alerts") { $(host).replaceChildren(); }

/* ---------- theme ---------- */
function applyTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); }
try { applyTheme(localStorage.getItem("theme")); } catch (e) { /* storage blocked */ }
$("theme-btn").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark"; applyTheme(next);
  try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
};

/* ---------- top-level tabs ---------- */
function showView(v) {
  for (const n of ["screen", "lab"]) {
    $("view-" + n).hidden = n !== v;
    $("tab-" + n).classList.toggle("active", n === v);
    $("tab-" + n).setAttribute("aria-selected", n === v);
  }
  if (v === "lab") initLab();
}
$("tab-screen").onclick = () => showView("screen");
$("tab-lab").onclick = () => showView("lab");
function showRight(which) {
  $("rp-criteria").hidden = which !== "criteria"; $("rp-summary").hidden = which !== "summary";
  $("rt-criteria").classList.toggle("active", which === "criteria"); $("rt-summary").classList.toggle("active", which === "summary");
  $("rt-criteria").setAttribute("aria-selected", which === "criteria"); $("rt-summary").setAttribute("aria-selected", which === "summary");
}
$("rt-criteria").onclick = () => showRight("criteria");
$("rt-summary").onclick = () => showRight("summary");

/* ---------- status ---------- */
function modelHeader(st) {
  const desc = (m) => `${m.name} · ${m.device} · ${m.dtype}`;
  if (!st.target || !st.draft) {   // older/simple backends: single model
    return st.model_name ? `${st.model_name} · ${st.device} · ${st.dtype}`
      : (st.phase === "loading" ? "loading model… (CPU load can take minutes)" : "no model");
  }
  const t = st.target, d = st.draft;
  const tgt = t.loaded ? `Target: ${desc(t)}` : (t.error ? "Target: failed to load" : "Target: loading… (CPU load can take minutes)");
  const drf = d.loaded ? `Draft: ${desc(d)}` : (d.loading ? "Draft: loading…" : (d.error ? "Draft: failed to load" : "Draft: not loaded"));
  const active = st.model_key === "small" && st.model_name ? `Active: ${st.model_name} · ` : "";
  return `${active}${tgt}  |  ${drf}`;
}

async function refreshStatus() {
  let st;
  try { st = await api("/api/status"); } catch (e) { setMode("error", "OFFLINE", e.message); schedule(5000); return; }
  S.status = st;
  const mode = st.mode === "demo" ? "demo" : (st.phase === "ready" ? "real" : (st.phase === "error" ? "error" : "loading"));
  const label = st.mode === "demo" ? "DEMO" : (st.phase === "ready" ? "REAL" : (st.phase === "error" ? "ERROR" : "LOADING"));
  setMode(mode, label, null);
  $("model-info").textContent = st.mode === "demo" ? "cached validation predictions · no model" : modelHeader(st);
  const ban = $("banner");
  const msgs = [];
  if (st.mode === "demo") msgs.push(st.demo_banner + ". Only notes and criteria from the validation set can be answered.");
  if (st.fallback_reason) msgs.push(st.fallback_reason);
  if (st.phase === "error" && st.mode === "real") msgs.push("Model failed to load: " + (st.message || "unknown error"));
  ban.hidden = msgs.length === 0; ban.textContent = msgs.join(" · ");
  ban.className = "banner" + (st.mode === "real" && st.phase === "loading" ? " info" : "");
  if (st.mode === "real" && st.phase === "loading") { ban.hidden = false; ban.className = "banner info"; ban.textContent = "Loading the model in the background… Run is disabled until it is ready."; }
  fillModelSelect(st); fillEngines(st);
  const busy = S.job && ["queued", "running"].includes(S.job.status);
  $("run-btn").disabled = busy || st.phase !== "ready";
  $("run-btn").title = st.phase !== "ready" ? "The model is not ready yet." : "";
  if (S.labReady) updateLabAvailability();
  schedule(st.phase === "loading" || Object.values(st.models || []).some((m) => m.loading) ? 1500 : 8000);
}
function setMode(cls, text) { const b = $("mode-badge"); b.className = "mode-badge mode-" + cls; b.textContent = text; }
function schedule(ms) { clearTimeout(S.stTimer); S.stTimer = setTimeout(refreshStatus, ms); }
function fillModelSelect(st) {
  const sel = $("model-sel"); const sig = JSON.stringify(st.models);
  if (sel.dataset.sig === sig) return; sel.dataset.sig = sig;
  sel.replaceChildren();
  if (st.mode === "demo" || !st.models.length) { sel.append(h("option", {}, "Cached predictions")); sel.disabled = true; return; }
  sel.disabled = false;
  for (const m of st.models) sel.append(h("option", { value: m.key, selected: m.active }, m.label + (m.loading ? " — loading…" : m.loaded ? "" : m.error ? " — failed" : " — not loaded (downloads on first use)")));
}
$("model-sel").onchange = async (e) => {
  clearAlerts();
  try { await post("/api/model", { key: e.target.value }); } catch (er) { alertBox(er.message); }
  refreshStatus();
};
function fillEngines(st) {
  const sel = $("engine-sel"); const sig = JSON.stringify(st.engines) + st.mode;
  if (sel.dataset.sig === sig) return; sel.dataset.sig = sig;
  const keep = sel.value; sel.replaceChildren();
  if (st.mode === "demo") { sel.append(h("option", {}, "cached (demo)")); sel.disabled = true; $("engine-warn").textContent = ""; return; }
  sel.disabled = false;
  for (const e of st.engines) sel.append(h("option", { value: e.id, disabled: !e.available, title: e.reason || "" }, e.label + (e.available ? "" : " (unavailable)")));
  const un = st.engines.find((e) => !e.available && e.reason && e.id === "speculative");
  if (un && $("engine-hint")) { $("engine-hint").textContent = "Speculative unavailable: " + un.reason; }
  else if ($("engine-hint")) { $("engine-hint").textContent = ""; }
  sel.value = (st.engines.find((e) => e.id === keep && e.available) ? keep : "kv");
  updateEngineWarn();
}
function updateEngineWarn() {
  const e = (S.status.engines || []).find((x) => x.id === $("engine-sel").value);
  $("engine-warn").textContent = e && e.warning ? "⚠ " + e.warning : "";
}
$("engine-sel").onchange = updateEngineWarn;

/* ---------- note ---------- */
function renderNote() {
  const box = $("sentences"); box.replaceChildren();
  if (!S.sentences.length) { box.append(h("p", { class: "muted" }, "No note loaded.")); return; }
  for (const s of S.sentences) {
    const el = h("div", { class: "sent", tabindex: 0, role: "button", "data-id": s.id, "aria-pressed": "false", id: "sent-" + s.id },
      h("span", { class: "sid" }, `[S${s.id}]`), h("span", {}, s.text));
    el.onclick = () => selectSentence(s.id);
    el.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); selectSentence(s.id); } };
    box.append(el);
  }
  refreshHighlights();
}
async function useNote(text, label) {
  clearAlerts();
  try {
    const r = await post("/api/note/prepare", { text });
    S.sentences = r.sentences; S.note = r.note;
    $("note-info").textContent = r.numbered_by_app ? r.warnings.join(" ") : `Note kept as numbered (${r.sentences.length} sentences).`;
    S.selSent = null; renderNote(); return true;
  } catch (e) { alertBox(e.message); return false; }
}
$("note-use").onclick = () => useNote($("note-text").value);
$("note-example").onchange = (e) => { const n = S.examples.notes.find((x) => x.id === e.target.value); if (n) { $("note-text").value = n.note; useNote(n.note); } };
async function readFile(input, cb) {
  const f = input.files[0]; input.value = ""; if (!f) return;
  try {
    const raw = await f.arrayBuffer();
    const r = await api("/api/upload", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: raw });
    cb(r.text);
  } catch (e) { alertBox(e.message); }
}
$("note-file").onchange = (e) => readFile(e.target, (t) => { $("note-text").value = t; $("paste-note").open = true; useNote(t); });
$("crit-file").onchange = (e) => readFile(e.target, (t) => { $("crit-text").value = t; parseCriteria(); });

/* ---------- criteria ---------- */
function renderCritRows() {
  const box = $("crit-rows"); box.replaceChildren();
  S.crit.forEach((c, i) => {
    const sel = h("select", { "aria-label": `Type of criterion ${i + 1}`, class: c.type ? "" : "need" },
      h("option", { value: "" }, "— choose type —"), h("option", { value: "inclusion", selected: c.type === "inclusion" }, "inclusion"),
      h("option", { value: "exclusion", selected: c.type === "exclusion" }, "exclusion"));
    sel.onchange = () => { c.type = sel.value || null; sel.classList.toggle("need", !c.type); };
    const ta = h("textarea", { rows: 2, "aria-label": `Text of criterion ${i + 1}` }); ta.value = c.text;
    ta.oninput = () => { c.text = ta.value; };
    box.append(h("div", { class: "crit-row" }, sel, ta, h("button", { class: "btn small-btn", title: "Remove", onclick: () => { S.crit.splice(i, 1); renderCritRows(); } }, "✕")));
  });
  $("crit-count").textContent = S.crit.length;
}
async function parseCriteria() {
  clearAlerts();
  try {
    const r = await post("/api/criteria/parse", { text: $("crit-text").value });
    S.crit = r.criteria.map((c) => ({ type: c.type, text: c.text })); renderCritRows();
    r.warnings.forEach((w) => alertBox(w, r.criteria.length ? "warn" : "error"));
  } catch (e) { alertBox(e.message); }
}
$("crit-parse").onclick = parseCriteria;
$("crit-add").onclick = () => { S.crit.push({ type: null, text: "" }); renderCritRows(); };
$("trial-example").onchange = async (e) => {
  const t = S.examples.trials.find((x) => x.id === e.target.value); if (!t) return;
  S.crit = t.criteria.map((c) => ({ type: c.type, text: c.text })); renderCritRows();
  const n = S.examples.notes.find((x) => x.id === t.patient_id);
  if (n && n.note !== $("note-text").value) { $("note-text").value = n.note; $("note-example").value = n.id; await useNote(n.note); }
  $("crit-text").value = "";
};

/* ---------- screening run ---------- */
$("run-btn").onclick = async () => {
  clearAlerts();
  if (!S.note && !$("note-text").value.trim()) { alertBox("The patient note is empty. Load an example or paste a note."); return; }
  if (!S.note) { if (!(await useNote($("note-text").value))) return; }
  if (!S.crit.length && $("crit-text").value.trim()) await parseCriteria();
  if (S.crit.some((c) => c.text.trim() && !c.type)) { alertBox("Choose a type (inclusion or exclusion) for every criterion."); return; }
  try {
    const r = await post("/api/screen", { note: S.note, criteria: S.crit, engine: $("engine-sel").value });
    r.warnings.forEach((w) => alertBox(w.message, "warn"));
    S.job = { id: r.job_id, status: "running" }; resetExports();
    $("criteria-input").open = false; $("results-title").hidden = false; showRight("criteria");
    $("cancel-btn").hidden = false; $("run-btn").disabled = true; poll();
  } catch (e) {
    alertBox(e.message);
    (e.details || []).slice(1).forEach((d) => alertBox(d.message));
  }
};
$("cancel-btn").onclick = async () => { if (S.job) { try { const r = await post(`/api/jobs/${S.job.id}/cancel`); alertBox(r.note, "info"); } catch (e) { alertBox(e.message); } } };
$("reset-btn").onclick = async () => {
  clearTimeout(S.pollTimer); try { await post("/api/reset"); } catch (e) { /* ignore */ }
  S.job = null; S.sentences = []; S.note = ""; S.crit = []; S.selSent = S.selCard = null;
  for (const id of ["note-text", "crit-text"]) $(id).value = "";
  $("note-example").value = ""; $("trial-example").value = "";
  $("note-info").textContent = ""; $("cards").replaceChildren(); $("results-title").hidden = true;
  $("progress").hidden = true; $("cancel-btn").hidden = true; $("criteria-input").open = true;
  $("summary").replaceChildren(h("p", { class: "muted" }, "Run a screening to see the preliminary outcome."));
  clearAlerts(); renderNote(); renderCritRows(); resetExports(); refreshStatus();
};
function resetExports() { for (const id of ["exp-json", "exp-html", "exp-md", "exp-print"]) { $(id).classList.add("disabled"); $(id).removeAttribute("href"); } }
function enableExports(id) {
  const m = { "exp-json": "json", "exp-html": "html", "exp-md": "md", "exp-print": "print" };
  for (const [el, f] of Object.entries(m)) { $(el).classList.remove("disabled"); $(el).href = `/api/jobs/${id}/export?format=${f}`; }
}
async function poll() {
  if (!S.job) return;
  let j;
  try { j = await api(`/api/jobs/${S.job.id}`); } catch (e) { alertBox(e.message); $("cancel-btn").hidden = true; return; }
  S.job = j; renderJob(j);
  if (["queued", "running"].includes(j.status)) { S.pollTimer = setTimeout(poll, 700); return; }
  $("cancel-btn").hidden = true; $("run-btn").disabled = !(S.status && S.status.phase === "ready");
  if (j.error) alertBox(j.error);
  if (j.save_error) alertBox(j.save_error, "warn");
  if (j.report_paths) alertBox("Report saved to " + j.report_paths.html.replace(/^.*outputs/, "outputs"), "info");
  enableExports(j.id); renderSummary(j.aggregate);
}
function renderJob(j) {
  const p = $("progress"); p.hidden = false;
  $("bar-fill").style.width = `${(100 * j.done) / Math.max(1, j.total)}%`;
  const cur = j.current !== null && j.current !== undefined ? j.current + 1 : j.done;
  let t = j.status === "running" ? `criterion ${Math.min(cur, j.total)}/${j.total} · elapsed ${fmt(j.elapsed_s, 0)} s` : `${j.status}: ${j.done}/${j.total} criteria · ${fmt(j.elapsed_s, 0)} s`;
  if (j.eta_s !== null && j.eta_s !== undefined) t += ` · est. remaining ~${fmt(j.eta_s, 0)} s`;
  $("progress-text").textContent = t;
  renderCards(j.results); refreshHighlights();
}
function renderCards(results) {
  const box = $("cards"); box.replaceChildren();
  for (const r of results) box.append(card(r));
}
function card(r) {
  if (r.status === "pending") return h("div", { class: "rcard pending", id: "card-" + r.index }, h("div", { class: "rhead" }, h("span", { class: `pill ${r.criterion_type}` }, r.criterion_type), h("span", { class: "muted" }, "queued…")), h("p", { class: "ctext" }, r.criterion_text));
  const b = r.badge;
  const el = h("div", { class: `rcard c-${b.color}`, id: "card-" + r.index, "data-cited": (r.evidence_ids || []).join(","), tabindex: 0 });
  el.append(h("div", { class: "rhead" },
    h("span", { class: `pill ${r.criterion_type}` }, r.criterion_type),
    h("span", { class: `badge ${b.color}` }, icon(b.icon), b.text, r.label ? h("code", {}, "· " + r.label) : null),
    r.missing_info ? h("span", { class: "flag" }, "Missing information") : null,
    r.flags.invalid ? h("span", { class: "flag bad" }, "Invalid output") : null,
    r.flags.error ? h("span", { class: "flag bad" }, "Error / unavailable") : null,
    r.banner ? h("span", { class: "demo-tag" }, "DEMO") : null,
    h("span", { class: "small muted mono", style: "margin-left:auto" }, r.elapsed_s != null ? `${fmt(r.elapsed_s, 1)} s` : "")));
  el.append(h("p", { class: "ctext" }, r.criterion_text));
  if (r.cited && r.cited.length) {
    el.append(h("div", { class: "cited" }, r.cited.map((c) => h("button", { class: "idpill", "data-id": c.id, title: "Show in note", onclick: () => { selectCard(r.index); selectSentence(c.id, true); } }, `[S${c.id}]`))));
    r.cited.forEach((c) => el.append(h("div", { class: "cite-text" }, h("span", { class: "mono muted" }, `[S${c.id}] `), c.text)));
  }
  el.append(h("p", { class: "expl" }, r.explanation.text));
  if (r.banner) el.append(h("div", { class: "small warn" }, r.banner));
  if (r.message && r.status !== "ok") el.append(h("div", { class: "small warn" }, r.message));
  el.append(h("details", { class: "drawer" }, h("summary", {}, "Details"),
    h("div", {}, `status: ${r.status} · parse: ${r.parse_error || (r.status === "ok" ? "ok" : "n/a")} · engine: ${r.engine || "n/a"} · latency: ${r.latency_s != null ? fmt(r.latency_s, 2) + " s" : "n/a"} · tokens: ${r.n_tokens ?? "n/a"}`),
    h("pre", { class: "mono" }, r.raw_output == null ? "(no raw output)" : r.raw_output)));
  el.addEventListener("click", (ev) => { if (!ev.target.closest("button,summary,details")) selectCard(r.index); });
  return el;
}

/* ---------- evidence sync ---------- */
function selectSentence(id, scroll) {
  S.selSent = S.selSent === id && !scroll ? null : id;
  refreshHighlights();
  const el = $("sent-" + id); if (el && scroll) el.scrollIntoView({ block: "center", behavior: "smooth" });
  if (el && scroll) el.focus({ preventScroll: true });
  if (S.selSent !== null) { // scroll to first citing card when the sentence was clicked in the note
    const first = document.querySelector(`.rcard.hl`); if (first && !scroll) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}
function selectCard(i) { S.selCard = S.selCard === i ? null : i; refreshHighlights(); }
function refreshHighlights() {
  const cardIds = S.selCard !== null && S.job && S.job.results[S.selCard] ? (S.job.results[S.selCard].evidence_ids || []) : [];
  document.querySelectorAll(".sent").forEach((el) => {
    const id = Number(el.dataset.id);
    el.classList.toggle("selected", id === S.selSent); el.classList.toggle("evidence", cardIds.includes(id));
    el.setAttribute("aria-pressed", id === S.selSent);
  });
  document.querySelectorAll(".rcard").forEach((el) => {
    const ids = (el.dataset.cited || "").split(",").filter(Boolean).map(Number);
    el.classList.toggle("hl", S.selSent !== null && ids.includes(S.selSent));
  });
  document.querySelectorAll(".idpill").forEach((el) => el.classList.toggle("on", Number(el.dataset.id) === S.selSent));
}

/* ---------- summary ---------- */
function renderSummary(a) {
  const box = $("summary"); box.replaceChildren();
  if (!a) { box.append(h("p", { class: "muted" }, "Run a screening to see the preliminary outcome.")); return; }
  box.append(h("div", { class: `outcome ${a.code}` }, a.outcome));
  box.append(h("p", { class: "small" }, a.rule_text));
  box.append(h("div", { class: "alert warn" }, a.disclaimer));
  a.notes.forEach((n) => box.append(h("p", { class: "small muted" }, n)));
  const c = a.counts;
  box.append(h("p", { class: "small mono" }, `decisive negative: ${c.decisive_negative} · unresolved: ${c.unresolved} · not applicable: ${c.not_applicable} · favourable: ${c.favourable} · skipped: ${c.skipped}`));
  if (a.contributing.length) {
    box.append(h("h3", {}, "Contributing criteria"));
    box.append(h("ul", {}, a.contributing.map((x) => h("li", {}, h("a", { href: "#card-" + x.index, onclick: (ev) => { ev.preventDefault(); showRight("criteria"); const el = $("card-" + x.index); if (el) { el.scrollIntoView({ block: "center" }); el.focus(); } } }, `#${x.index + 1}`), ` (${x.criterion_type}) ${x.text}: `, h("em", {}, x.reason || x.label)))));
  }
  if (a.unresolved.length && a.code === "ineligible") {
    box.append(h("h3", {}, "Also unresolved"));
    box.append(h("ul", {}, a.unresolved.map((x) => h("li", {}, `#${x.index + 1} (${x.criterion_type}) ${x.text}: ${x.reason}`))));
  }
}

/* ---------- init ---------- */
async function init() {
  renderCritRows();
  await refreshStatus();
  try {
    S.examples = await api("/api/examples");
    for (const n of S.examples.notes) $("note-example").append(h("option", { value: n.id }, n.label));
    for (const t of S.examples.trials) $("trial-example").append(h("option", { value: t.id }, t.label));
    if (!S.examples.notes.length) {
      const msg = S.examples.error || "No validation examples found. Run: python scripts/prepare_data.py";
      $("note-info").textContent = msg + " You can still paste your own note.";
      for (const id of ["note-example", "trial-example"]) {
        const sel = $(id); sel.append(h("option", { value: "", disabled: true }, "Run python scripts/prepare_data.py"));
        sel.title = msg;
      }
    }
  } catch (e) { alertBox(e.message); }
}

/* =========================================================== Inference Lab */
const ECOL = { plain: "var(--c-plain)", kv: "var(--c-kv)", speculative: "var(--c-spec)" };
const ENAME = { plain: "Plain (no cache)", kv: "KV-cached", speculative: "Speculative" };
function initLab() {
  if (S.labReady) return; S.labReady = true;
  loadSaved(); loadPrompts(); drawAttention(); updateLabAvailability();
}
async function loadPrompts() {
  try { const r = await api("/api/lab/prompts"); const s = $("lab-val"); r.prompts.forEach((p) => s.append(h("option", { value: p.id }, p.label)));
    if (!r.prompts.length) s.append(h("option", { value: "", disabled: true }, "Run python scripts/prepare_data.py")); } catch (e) { /* optional */ }
}
function updateLabAvailability() {
  const st = S.status || {}; const live = st.live_available && st.phase === "ready";
  const models = Object.fromEntries((st.models || []).map((m) => [m.key, m]));
  document.querySelectorAll('input[name="eng"]').forEach((cb) => {
    const spec = cb.value === "speculative"; const m = $("lab-model").value;
    const ok = live && (!spec || (m === "merged" && models.small && models.small.loaded));
    cb.disabled = !ok; if (!ok) cb.checked = false;
    cb.parentElement.title = ok ? "" : (!live ? "Live runs need real mode with a loaded model." : "Needs the merged 4B target and the 0.6B draft loaded.");
  });
  $("bench-btn").disabled = $("stream-btn").disabled = !live;
  $("lab-status").textContent = live ? "" : (st.mode === "demo" ? "Demo mode: live runs are disabled (no model)." : "Model not ready.");
  const small = models.small;
  if (live && small && !small.loaded && !small.loading) $("lab-status").textContent = "Tip: select the 0.6B model in the header model selector to load it (needed for speculative and the fast CPU option).";
  if (S.status && S.status.lab) { if (!S.labInit) { $("lab-n").value = S.status.lab.default_new_tokens; $("lab-k").value = S.status.lab.default_k; S.labInit = true; } }
}
$("lab-model").onchange = updateLabAvailability;
$("lab-prompt-src").onchange = () => { $("lab-val-wrap").hidden = $("lab-prompt-src").value !== "val"; };

/* ---- charts (inline SVG) ---- */
function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { hi = lo + 1; }
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const out = []; for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}
function shapeEl(kind, x, y, color, title) {
  const r = 4; let e;
  if (kind === "square") e = sv("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, fill: color });
  else if (kind === "diamond") e = sv("path", { d: `M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z`, fill: color });
  else e = sv("circle", { cx: x, cy: y, r, fill: color });
  e.append(sv("title", {}, title)); return e;
}
function lineChart(host, o) {
  const W = 520, H = 280, L = 52, R = 14, T = 30, B = 62;
  const pts = o.series.flatMap((s) => s.points).filter((p) => p[1] !== null && p[1] !== undefined);
  host.replaceChildren(h("div", { class: "small muted" }, o.title));
  if (!pts.length) { host.append(h("p", { class: "muted small" }, "No data.")); return; }
  const xs = [...new Set(pts.map((p) => p[0]))].sort((a, b) => a - b);
  let ymax = Math.max(...pts.map((p) => p[1])), ymin = o.yZero ? 0 : Math.min(...pts.map((p) => p[1]));
  const yt = niceTicks(ymin, ymax); ymin = yt[0]; ymax = yt[yt.length - 1];
  const lx = (x) => o.xLog ? Math.log2(x) : x;
  const x0 = lx(xs[0]), x1 = lx(xs[xs.length - 1]) === x0 ? x0 + 1 : lx(xs[xs.length - 1]);
  const X = (x) => L + ((lx(x) - x0) / (x1 - x0)) * (W - L - R), Y = (y) => H - B - ((y - ymin) / (ymax - ymin || 1)) * (H - B - T);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title });
  svg.append(sv("text", { class: "t", x: L, y: 16 }, o.title));
  yt.forEach((v) => { svg.append(sv("line", { class: "grid", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }), sv("text", { x: L - 6, y: Y(v) + 3, "text-anchor": "end" }, +v.toPrecision(3))); });
  xs.forEach((v) => svg.append(sv("text", { x: X(v), y: H - B + 14, "text-anchor": "middle" }, v)));
  svg.append(sv("text", { x: (L + W - R) / 2, y: H - B + 28, "text-anchor": "middle" }, o.xLabel));
  svg.append(sv("text", { x: 12, y: (T + H - B) / 2, transform: `rotate(-90 12 ${(T + H - B) / 2})`, "text-anchor": "middle" }, o.yLabel));
  const shapes = ["circle", "square", "diamond"];
  o.series.forEach((s, i) => {
    const p = s.points.filter((q) => q[1] !== null && q[1] !== undefined).sort((a, b) => a[0] - b[0]);
    svg.append(sv("polyline", { points: p.map((q) => `${X(q[0])},${Y(q[1])}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }));
    p.forEach((q) => svg.append(shapeEl(s.shape || shapes[i % 3], X(q[0]), Y(q[1]), s.color, `${s.name}: ${o.xLabel} ${q[0]} → ${fmt(q[1], 3)}`)));
  });
  let lxp = L; // legend below axis (colour + marker shape + dash, not colour alone)
  o.series.forEach((s, i) => {
    const y = H - 10; svg.append(sv("line", { x1: lxp, x2: lxp + 18, y1: y - 3, y2: y - 3, stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }));
    svg.append(shapeEl(s.shape || shapes[i % 3], lxp + 9, y - 3, s.color, s.name)); svg.append(sv("text", { x: lxp + 22, y }, s.name)); lxp += 30 + s.name.length * 5.4;
    if (lxp > W - 80 && i < o.series.length - 1) lxp = L;
  });
  host.append(svg);
}

/* ---- saved results ---- */
async function loadSaved() {
  const root = $("saved-root");
  let d; try { d = await api("/api/lab/saved"); } catch (e) { root.replaceChildren(h("div", { class: "alert error" }, e.message)); return; }
  S.saved = d; renderSaved();
}
function provLine(sec) { return h("div", { class: "prov" }, sec.provenance.label); }
function renderSaved() {
  const d = S.saved, root = $("saved-root"); root.replaceChildren();
  const kv = d.kv_cache, sp = d.speculative, at = d.attention;
  for (const [name, sec] of [["KV-cache benchmark", kv], ["Speculative benchmark", sp], ["Attention benchmark", at]]) {
    if (!sec.available) root.append(h("div", { class: "alert info" }, `${name}: not available (${sec.reason}).`));
  }
  const lens = (rows, f) => [...new Set(rows.filter((r) => r.source === "synthetic").map(f))].sort((a, b) => a - b);
  const rows = (kv.available ? kv.cases : []).concat([]);
  const inLens = kv.available ? lens(kv.cases, (r) => r.input_len) : (sp.available ? lens(sp.rows, (r) => r.input_len) : []);
  const outLens = kv.available ? lens(kv.cases, (r) => r.out_len) : (sp.available ? lens(sp.rows, (r) => r.out_len) : []);
  if (inLens.length) {
    const si = h("select", { id: "sv-in" }, inLens.map((x) => h("option", { value: x, selected: x === 512 }, x)));
    const so = h("select", { id: "sv-out" }, outLens.map((x) => h("option", { value: x, selected: x === 64 }, x)));
    si.onchange = so.onchange = renderSavedBody;
    root.append(h("div", { class: "row" }, h("label", {}, "Input length (tokens)", si), h("label", {}, "Output length (tokens)", so),
      h("span", { class: "small muted" }, "Synthetic prompts; medians over repeats.")));
  }
  root.append(h("div", { id: "sv-body" }));
  renderSavedBody();
}
function renderSavedBody() {
  const d = S.saved, body = $("sv-body"); body.replaceChildren();
  const kv = d.kv_cache, sp = d.speculative, at = d.attention;
  const iL = Number(($("sv-in") || {}).value), oL = Number(($("sv-out") || {}).value);
  const cards = h("div", { class: "grid3" });
  const kvc = kv.available ? kv.cases.find((c) => c.source === "synthetic" && c.input_len === iL && c.out_len === oL) : null;
  const mk = (name, color, items) => h("div", { class: "ecard", style: `--c:${color}` }, h("h4", {}, name), h("dl", {}, items.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
  if (kvc && kvc.uncached) cards.append(mk("Plain (no cache)", ECOL.plain, [["prefill", fmt(kvc.uncached.prefill_s, 3) + " s"], ["decode tok/s", fmt(kvc.uncached.decode_tok_s, 1)], ["peak mem Δ", fmt(kvc.uncached.mem_delta_mib, 0) + " MiB"]]));
  if (kvc && kvc.cached) cards.append(mk("KV-cached", ECOL.kv, [["prefill", fmt(kvc.cached.prefill_s, 3) + " s"], ["decode tok/s", fmt(kvc.cached.decode_tok_s, 1)], ["peak mem Δ", fmt(kvc.cached.mem_delta_mib, 0) + " MiB"]]));
  if (sp.available) sp.rows.filter((r) => r.source === "synthetic" && r.decoder === "speculative" && r.input_len === iL && r.out_len === oL).sort((a, b) => a.k - b.k).forEach((r) =>
    cards.append(mk(`Speculative k=${r.k}`, ECOL.speculative, [["prefill", fmt(r.prefill_s, 3) + " s"], ["decode tok/s", fmt(r.decode_tok_s, 1)], ["peak mem (abs)", fmt(r.peak_mem_mib, 0) + " MiB"],
      ["acceptance", fmt(r.acceptance_rate * 100, 0) + " %"], ["speedup vs plain cached", fmt(r.speedup, 2) + "×"], ["tokens = plain", r.tokens_equal ? "yes" : "NO"]])));
  if (!cards.children.length) cards.append(h("p", { class: "muted" }, "No saved row for this input/output length."));
  body.append(cards);
  if (kv.available) body.append(provLine(kv)); if (sp.available) body.append(provLine(sp));
  const charts = h("div", { class: "charts" });
  const c1 = h("div", { class: "chart" }), c2 = h("div", { class: "chart" }), c3 = h("div", { class: "chart" }), c4 = h("div", { class: "chart" });
  charts.append(c1, c2, c3, c4); body.append(charts);
  if (kv.available) {
    const sel = (k) => kv.cases.filter((c) => c.source === "synthetic" && c.out_len === oL && c[k]).map((c) => [c.input_len, c[k].tok_s]);
    lineChart(c1, { title: `Tokens/s vs input length (output ${oL}) — saved`, xLabel: "input tokens", yLabel: "tokens/s", yZero: true, xLog: true,
      series: [{ name: "cached", color: ECOL.kv, points: sel("cached") }, { name: "uncached", color: ECOL.plain, points: sel("uncached"), dash: "5 3" }] });
  } else c1.append(h("p", { class: "muted small" }, "KV-cache data unavailable."));
  if (sp.available) {
    const ks = [...new Set(sp.rows.filter((r) => r.decoder === "speculative").map((r) => r.k))].sort((a, b) => a - b);
    const ins = lens(sp.rows, (r) => r.input_len); const cols = ["#6366F1", "#8B5CF6", "#A78BFA"];
    const ser = ins.map((n, i) => ({ name: `in ${n}`, color: cols[i % 3], points: sp.rows.filter((r) => r.source === "synthetic" && r.decoder === "speculative" && r.out_len === oL && r.input_len === n).map((r) => [r.k, r.speedup]) }));
    ser.push({ name: "plain = 1×", color: ECOL.plain, dash: "3 3", points: ks.map((k) => [k, 1]), shape: "circle" });
    lineChart(c2, { title: `Speculative speedup vs k (output ${oL}) — saved`, xLabel: "k (draft tokens)", yLabel: "speedup vs plain cached", series: ser, yZero: true });
  } else c2.append(h("p", { class: "muted small" }, "Speculative data unavailable."));
  if (at.available && at.results.length) {
    const impls = [...new Set(at.results.filter((r) => r.status === "ok").map((r) => r.impl))]; const pal = ["#0284C7", "#06B6D4", "#6366F1", "#D97706", "#059669", "#DC2626", "#64748B", "#8B5CF6"];
    const mkS = (key) => impls.map((im, i) => ({ name: im, color: pal[i % pal.length], dash: i % 2 ? "5 3" : "", points: at.results.filter((r) => r.impl === im && r.status === "ok").map((r) => [r.seq_len, r[key]]) }));
    lineChart(c3, { title: "Attention time vs sequence length — saved", xLabel: "sequence length", yLabel: "ms per call", xLog: true, series: mkS("time_ms") });
    lineChart(c4, { title: "Attention peak extra memory vs sequence length — saved", xLabel: "sequence length", yLabel: "MiB", xLog: true, series: mkS("peak_mem_mib") });
    body.append(provLine(at));
  } else { c3.append(h("p", { class: "muted small" }, at.available ? "No attention rows." : "Attention benchmark file is not configured or not found (see benchmarks.attention in configs/app.yaml).")); }
  const eq = h("div", { class: "alert info" });
  const lines = [];
  if (kv.available) { const e = kv.equality; lines.push(`KV-cache vs uncached: ${e.n_equal}/${e.n_cases} cases token-identical.`); e.mismatches.forEach((m) => lines.push(`  mismatch: input ${m.input_len}, output ${m.out_len} (${m.source}) — ${typeof m.detail === "string" ? m.detail : JSON.stringify(m.detail)}`)); }
  if (sp.available) { const e = sp.equality; lines.push(`Speculative vs plain: ${e.n_equal}/${e.n_rows} rows token-identical.`); e.mismatches.forEach((m) => lines.push(`  mismatch: input ${m.input_len}, output ${m.out_len}, k=${m.k} — ${typeof m.detail === "string" ? m.detail : JSON.stringify(m.detail)}`)); }
  if (lines.length) { lines.push("Mismatches are reported, not hidden: in fp16 two tokens can have (near-)equal logits, so a different kernel path can flip the argmax. Check the logit gap in each line."); eq.append(h("pre", { class: "mono small", style: "white-space:pre-wrap;margin:0" }, lines.join("\n"))); body.append(h("h3", {}, "Token equality"), eq); }
}

/* ---- live bench / stream ---- */
function selectedEngines() { return [...document.querySelectorAll('input[name="eng"]:checked')].map((c) => c.value); }
async function startLab(kind) {
  clearAlerts("lab-alerts");
  const engines = selectedEngines(); if (!engines.length) { alertBox("Select at least one engine.", "error", "lab-alerts"); return; }
  const src = $("lab-prompt-src").value; let prompt;
  if (src === "val") prompt = { source: "val", annotation_id: Number($("lab-val").value) };
  else {
    const first = S.crit.find((c) => c.text.trim() && c.type);
    if (!S.note || !first) { alertBox("Load a note and add at least one typed criterion in the Screening workspace first.", "error", "lab-alerts"); return; }
    prompt = { source: "current", note: S.note, criterion_type: first.type, criterion_text: first.text };
  }
  try {
    const r = await post("/api/lab/run", { kind, engines, model: $("lab-model").value, prompt, max_new_tokens: Number($("lab-n").value), k: Number($("lab-k").value) });
    S.labJob = r.job_id; $("live-results").replaceChildren(); buildPanes(engines, kind === "stream");
    $("lab-cancel").hidden = false; $("bench-btn").disabled = $("stream-btn").disabled = true; $("lab-status").textContent = "running…";
    listen(r.job_id, kind);
  } catch (e) { alertBox(e.message, "error", "lab-alerts"); }
}
$("bench-btn").onclick = () => startLab("bench");
$("stream-btn").onclick = () => startLab("stream");
$("lab-cancel").onclick = async () => { if (S.labJob) { try { await post(`/api/lab/jobs/${S.labJob}/cancel`); } catch (e) { /* ignore */ } } };
function buildPanes(engines, stream) {
  const box = $("stream-panes"); box.replaceChildren(); $("stream-legend").hidden = !stream; S.panes = {};
  for (const e of ["plain", "kv", "speculative"].filter((x) => engines.includes(x))) {
    const body = h("div", { class: "body" }, stream ? "" : "(benchmark run: tokens are not streamed)"), rate = h("span", { class: "mono small" }, "—");
    box.append(h("div", { class: "spane", style: `--c:${ECOL[e]}` }, h("div", { class: "hd" }, h("span", {}, ENAME[e]), rate), body)); S.panes[e] = { body, rate, n: 0 };
  }
}
function listen(id, kind) {
  if (S.es) S.es.close();
  const es = S.es = new EventSource(`/api/lab/jobs/${id}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data), p = S.panes[ev.engine];
    if (ev.type === "token" && p) {
      p.n = ev.n; p.body.append(h("span", { class: `tok tok-${ev.kind}`, title: ev.kind }, ev.text.replace(/\n/g, "↵\n")));
      p.rate.textContent = `${ev.n} tok · ${fmt(ev.n / Math.max(ev.t, 1e-3), 1)} tok/s`;
    } else if (ev.type === "engine_start" && p) { p.rate.textContent = "running…"; }
    else if (ev.type === "engine_end" && p) { p.rate.textContent = `${ev.n_tokens} tok · ${fmt(ev.tok_s, 1)} tok/s (done)`; }
    else if (ev.type === "error") { alertBox(ev.message, "error", "lab-alerts"); }
    else if (ev.type === "done") { es.close(); finishLab(id); }
  };
  es.onerror = () => { es.close(); if (S.labJob === id && $("lab-cancel").hidden === false) { alertBox("Lost the connection to the stream.", "warn", "lab-alerts"); finishLab(id); } };
}
async function finishLab(id) {
  $("lab-cancel").hidden = true; $("lab-status").textContent = ""; updateLabAvailability();
  let j; try { j = await api(`/api/lab/jobs/${id}`); } catch (e) { return; }
  const meta = j.results.meta; if (!meta) return;
  const order = ["plain", "kv", "speculative"].filter((e) => j.results[e]); const box = $("live-results"); box.replaceChildren();
  box.append(h("div", { class: "prov live" }, `Live results — ${meta.label} · ${meta.model} · prompt ${meta.prompt_tokens} tokens · max ${meta.max_new_tokens} new tokens (not the saved T4 numbers)`));
  const th = ["Engine", "Prefill (s)", "Decode tok/s", "Total tok/s", "Peak memory", "Acceptance", "Speedup vs plain", "Tokens = plain"];
  box.append(h("table", { class: "t" }, h("tr", {}, th.map((x) => h("th", {}, x))), order.map((e) => { const r = j.results[e]; const m = r.memory;
    return h("tr", {}, h("td", {}, ENAME[e]), h("td", { class: "n" }, fmt(r.prefill_s, 3)), h("td", { class: "n" }, fmt(r.decode_tok_s, 2)), h("td", { class: "n" }, fmt(r.tok_s, 2)),
      h("td", { class: "n" }, m ? `${fmt(m.delta_mib, 0)} MiB Δ ${m.kind.toUpperCase()} (peak ${fmt(m.peak_mib, 0)})` : "n/a (install psutil)"),
      h("td", { class: "n" }, r.acceptance_rate == null ? "—" : fmt(r.acceptance_rate * 100, 0) + " %"),
      h("td", { class: "n" }, r.speedup_vs_plain == null ? "—" : fmt(r.speedup_vs_plain, 2) + "×"),
      h("td", {}, r.tokens_equal_to_plain == null ? "n/a (plain not run)" : (r.tokens_equal_to_plain ? "identical" : "DIFFERENT")));
  })));
  if (order.some((e) => j.results[e].stream_consistent === false)) box.append(h("div", { class: "alert warn" }, "The streamed token view differed from the decoder's final tokens for one engine; the table uses the decoder's own output."));
}

/* ---- tiled attention ---- */
let atTimer = null;
function scheduleAttention() { clearTimeout(atTimer); atTimer = setTimeout(drawAttention, 150); }
$("at-seq").oninput = () => { $("at-seq-v").textContent = $("at-seq").value; $("at-row").max = $("at-seq").value - 1; scheduleAttention(); };
for (const id of ["at-block", "at-row", "at-causal"]) $(id).onchange = scheduleAttention;
async function drawAttention() {
  const seq = Number($("at-seq").value), block = Math.min(Number($("at-block").value), seq), row = Math.min(Number($("at-row").value), seq - 1);
  const out = $("at-out"); let d;
  try { d = await api(`/api/lab/attention?seq=${seq}&block=${block}&row=${row}&causal=${$("at-causal").checked}`); } catch (e) { out.replaceChildren(h("div", { class: "alert error" }, e.message)); return; }
  const cell = Math.max(14, Math.min(44, Math.floor(360 / d.n_blocks))), pad = 24, Wd = pad + cell * d.n_blocks + 4;
  const svg = sv("svg", { viewBox: `0 0 ${Wd} ${pad + cell * d.n_blocks + 4}`, width: Wd, class: "at-grid", role: "img", "aria-label": "Block grid: query blocks by key blocks" });
  const fillOf = { full: "#06B6D4", partial: "#6366F1", skipped: "transparent" };
  d.grid.forEach((line, qb) => line.forEach((st, kb) => {
    const g = sv("rect", { x: pad + kb * cell, y: pad + qb * cell, width: cell - 2, height: cell - 2, rx: 3, fill: fillOf[st], "fill-opacity": st === "skipped" ? 0 : 0.55,
      stroke: qb === d.query_block ? "#D97706" : "var(--border)", "stroke-width": qb === d.query_block ? 2 : 1, "stroke-dasharray": st === "skipped" ? "3 2" : "" });
    g.append(sv("title", {}, `query block ${qb} × key block ${kb}: ${st}`)); svg.append(g);
    svg.append(sv("text", { x: pad + kb * cell + cell / 2 - 1, y: pad + qb * cell + cell / 2 + 3, "text-anchor": "middle" }, st === "skipped" ? "×" : st === "partial" ? "◩" : "■"));
  }));
  d.grid.forEach((_, i) => { svg.append(sv("text", { x: pad - 4, y: pad + i * cell + cell / 2 + 3, "text-anchor": "end" }, i), sv("text", { x: pad + i * cell + cell / 2, y: pad - 6, "text-anchor": "middle" }, i)); });
  const rows = d.steps.map((s) => h("tr", {}, h("td", {}, s.key_block), h("td", {}, s.status === "skipped" ? "skipped (causal)" : s.status), h("td", { class: "n" }, s.visible_keys ?? "—"),
    h("td", { class: "n" }, s.running_max == null ? (s.status === "skipped" ? "—" : "−∞") : s.running_max.toFixed(4)), h("td", { class: "n" }, s.alpha == null ? "—" : s.alpha.toFixed(4)), h("td", { class: "n" }, s.normalizer == null ? "—" : s.normalizer.toFixed(4))));
  out.replaceChildren(h("div", { class: "row wrap", style: "align-items:flex-start;gap:24px" },
    h("div", {}, h("div", { class: "small muted" }, "Rows = query blocks, columns = key blocks. ■ full · ◩ partial (diagonal) · × skipped by the causal mask. Amber outline = block row of the chosen query."), svg),
    h("div", { style: "flex:1;min-width:320px" }, h("h3", {}, `Online softmax for query row ${d.row} (query block ${d.query_block})`),
      h("table", { class: "t" }, h("tr", {}, ["key block", "status", "visible keys", "running max m", "rescale α", "normalizer l"].map((x) => h("th", {}, x))), rows),
      h("p", { class: "mono small" }, `max |tiled − reference| = ${d.max_abs_err.toExponential(2)} · trace vs tiled row = ${d.trace_vs_tiled_err.toExponential(2)} · skipped ${d.n_skipped}/${d.n_blocks_total} blocks · reference ${d.ms_reference.toFixed(2)} ms, tiled ${d.ms_tiled.toFixed(2)} ms (CPU, ${d.head_dim}-dim head)`))));
}

init();
