"""Report export (pure): JSON structure, self-contained HTML (inline CSS, printable), Markdown, and saving to disk.

All user-supplied text is HTML-escaped. No PDF dependency: the HTML has a Print / Save as PDF button.
"""
import html
import json
import re
import time
from pathlib import Path

from trialmatch.app.config import DISCLAIMER

VALIDATION_STATEMENT = ("The underlying model was validated on criterion-level labels only (label accuracy and "
                        "evidence on the held-out validation split). The patient-level outcome shown here is a "
                        "rule-based aggregation and was not validated against expert patient-trial verdicts.")


def build_report(run: dict) -> dict:
    """run: the screening run record built by the job runner. Adds disclaimer, provenance and the missing-info list."""
    results = run.get("results", [])
    missing = [{"index": r["index"], "criterion_type": r["criterion_type"], "text": r["criterion_text"]}
               for r in results if r.get("missing_info")]
    gen = run.get("generation", {"greedy": True, "max_new_tokens": 64})
    return {
        "disclaimer": DISCLAIMER,
        "created_at": run.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run.get("run_id"),
        "mode": run.get("mode"),
        "backend": run.get("backend", {}),
        "engine": run.get("engine"),
        "note": run.get("note"),
        "sentences": run.get("sentences", []),
        "criteria": [{"index": r["index"], "type": r["criterion_type"], "text": r["criterion_text"]} for r in results],
        "results": results,
        "missing_information": missing,
        "aggregate": run.get("aggregate"),
        "warnings": run.get("warnings", []),
        "provenance": {
            "git_commit": run.get("git_commit"),
            "adapter_or_hub_id": run.get("backend", {}).get("adapter"),
            "decoding": f"{'greedy' if gen.get('greedy', True) else 'sampling'}, max_new_tokens {gen.get('max_new_tokens')}",
            "validation_statement": VALIDATION_STATEMENT,
            "explanations": "Template-built from the label and cited sentences only; no model-written explanations.",
        },
    }


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


_CSS = """
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#0f172a;background:#fff}
main{max-width:900px;margin:0 auto;padding:16px 20px 40px}
.disc{background:#FEF3C7;color:#92400E;padding:10px 20px;font-weight:600;border-bottom:1px solid #f59e0b}
h1,h2,h3{letter-spacing:-.02em;font-weight:600}h2{margin-top:28px;border-bottom:1px solid #e2e8f0;padding-bottom:4px}
code,.mono{font-family:ui-monospace,"JetBrains Mono",Consolas,monospace}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #e2e8f0;padding:5px 8px;text-align:left;vertical-align:top}
.sid{font-family:ui-monospace,Consolas,monospace;background:#f1f5f9;border-radius:4px;padding:0 5px}
.card{border:1px solid #e2e8f0;border-left:4px solid #94a3b8;border-radius:6px;padding:8px 12px;margin:10px 0}
.card.met{border-left-color:#059669}.card.unmet{border-left-color:#DC2626}.card.unknown{border-left-color:#D97706}
.pill{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600}
.pill.inclusion{background:#ECFDF5;color:#047857}.pill.exclusion{background:#FEF2F2;color:#B91C1C}
.badge{font-weight:600}.muted{color:#64748b}.print{margin:12px 0;padding:6px 14px}
@media print{.print{display:none}}
"""
_ICON = {"check": "✓", "x": "✗", "question": "?", "slash": "∅", "warning": "⚠"}


def render_html(rep: dict) -> str:
    p, b = rep["provenance"], rep.get("backend", {})
    agg = rep.get("aggregate") or {}
    o = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Trial screening report</title>"
         f"<style>{_CSS}</style></head><body><div class='disc'>{_e(rep['disclaimer'])}</div><main>",
         "<h1>Trial eligibility screening report</h1>",
         "<button class='print' onclick='window.print()'>Print / Save as PDF</button>"]
    if rep.get("mode") == "demo":
        o.append("<p><strong>DEMO MODE: cached validation prediction, not live inference.</strong></p>")
    o.append("<h2>Run details</h2><table>")
    for k, v in [("Timestamp", rep["created_at"]), ("Mode", rep.get("mode")), ("Model", b.get("model_name")),
                 ("Device / dtype", f"{b.get('device')} / {b.get('dtype')}"), ("Engine", rep.get("engine")),
                 ("Git commit", p.get("git_commit")), ("Adapter / Hub id", p.get("adapter_or_hub_id")),
                 ("Run id", rep.get("run_id"))]:
        o.append(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>")
    o.append("</table><h2>Overall preliminary outcome</h2>")
    o.append(f"<p class='badge'>{_e(agg.get('outcome'))}</p><p class='muted'>{_e(agg.get('rule_text'))}</p>"
             f"<p><strong>{_e(agg.get('disclaimer'))}</strong></p>")
    if agg.get("contributing"):
        o.append("<p>Contributing criteria:</p><ul>" + "".join(
            f"<li>#{c['index'] + 1} ({_e(c['criterion_type'])}): {_e(c['text'])} &mdash; {_e(c.get('reason', c.get('label')))}</li>"
            for c in agg["contributing"]) + "</ul>")
    for n in agg.get("notes", []):
        o.append(f"<p class='muted'>{_e(n)}</p>")
    o.append("<h2>Patient note (as numbered for the model)</h2><table>")
    for s in rep["sentences"]:
        o.append(f"<tr><td style='width:60px'><span class='sid'>[S{s['id']}]</span></td><td>{_e(s['text'])}</td></tr>")
    o.append("</table><h2>Per-criterion results</h2>")
    for r in rep["results"]:
        bd = r.get("badge", {})
        o.append(f"<div class='card {_e(bd.get('key'))}'><span class='pill {_e(r['criterion_type'])}'>"
                 f"{_e(r['criterion_type'])}</span> <strong>#{r['index'] + 1}</strong> {_e(r['criterion_text'])}<br>"
                 f"<span class='badge'>{_ICON.get(bd.get('icon'), '')} {_e(bd.get('text'))}"
                 f"{(' &mdash; <code>' + _e(r.get('label')) + '</code>') if r.get('label') else ''}</span>")
        if r.get("cited"):
            o.append("<ul>" + "".join(f"<li><span class='sid'>[S{c['id']}]</span> {_e(c['text'])}</li>"
                                      for c in r["cited"]) + "</ul>")
        o.append(f"<p>{_e(r.get('explanation', {}).get('text'))}</p>")
        if r.get("missing_info"):
            o.append("<p><strong>Missing information flagged.</strong></p>")
        o.append(f"<p class='muted'>status: {_e(r.get('status'))} &middot; engine: {_e(r.get('engine'))} &middot; "
                 f"latency: {_e(r.get('latency_s'))} s &middot; tokens: {_e(r.get('n_tokens'))}"
                 f"{' &middot; parse error: ' + _e(r.get('parse_error')) if r.get('parse_error') else ''}</p></div>")
    o.append("<h2>Missing information</h2>")
    mi = rep["missing_information"]
    o.append("<ul>" + "".join(f"<li>#{m['index'] + 1} ({_e(m['criterion_type'])}): {_e(m['text'])}</li>" for m in mi)
             + "</ul>" if mi else "<p>None flagged.</p>")
    if rep.get("warnings"):
        o.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{_e(w.get('message', w))}</li>" for w in rep["warnings"]) + "</ul>")
    o.append(f"<h2>Provenance</h2><ul><li>Decoding: {_e(p['decoding'])}</li><li>{_e(p['explanations'])}</li>"
             f"<li>{_e(p['validation_statement'])}</li></ul></main></body></html>")
    return "".join(o)


def render_markdown(rep: dict) -> str:
    p, b, agg = rep["provenance"], rep.get("backend", {}), rep.get("aggregate") or {}
    L = [f"> {rep['disclaimer']}", "", "# Trial eligibility screening report", ""]
    if rep.get("mode") == "demo":
        L += ["**DEMO MODE: cached validation prediction, not live inference.**", ""]
    L += [f"- Timestamp: {rep['created_at']}", f"- Mode: {rep.get('mode')}",
          f"- Model: {b.get('model_name')} ({b.get('device')}, {b.get('dtype')})", f"- Engine: {rep.get('engine')}",
          f"- Git commit: {p.get('git_commit')}", f"- Adapter / Hub id: {p.get('adapter_or_hub_id')}", "",
          "## Overall preliminary outcome", "", f"**{agg.get('outcome')}**", "", agg.get("rule_text", ""), "",
          f"**{agg.get('disclaimer', '')}**", "", "## Patient note", ""]
    L += [f"- `[S{s['id']}]` {s['text']}" for s in rep["sentences"]]
    L += ["", "## Per-criterion results", ""]
    for r in rep["results"]:
        bd = r.get("badge", {})
        L += [f"### {r['index'] + 1}. [{r['criterion_type']}] {r['criterion_text']}", "",
              f"- Result: {bd.get('text')}" + (f" (`{r['label']}`)" if r.get("label") else ""),
              f"- Explanation: {r.get('explanation', {}).get('text')}",
              f"- Status: {r.get('status')}; engine {r.get('engine')}; latency {r.get('latency_s')} s; tokens {r.get('n_tokens')}"]
        if r.get("missing_info"):
            L.append("- Missing information flagged")
        L.append("")
    L += ["## Missing information", ""]
    L += [f"- #{m['index'] + 1} ({m['criterion_type']}): {m['text']}" for m in rep["missing_information"]] or ["None flagged."]
    L += ["", "## Provenance", "", f"- Decoding: {p['decoding']}", f"- {p['explanations']}", f"- {p['validation_statement']}", ""]
    return "\n".join(L)


def save_report(rep: dict, reports_dir: Path) -> dict:
    """Write JSON + HTML with a timestamped name; returns the paths. Raises OSError on failure (caller records it)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9A-Za-z_-]", "", time.strftime("%Y%m%d_%H%M%S")) + f"_{rep.get('mode')}_{rep.get('run_id', '')}"
    jp, hp = reports_dir / f"report_{stamp}.json", reports_dir / f"report_{stamp}.html"
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    hp.write_text(render_html(rep), encoding="utf-8")
    return {"json": str(jp), "html": str(hp)}
