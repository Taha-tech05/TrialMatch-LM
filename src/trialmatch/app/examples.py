"""VAL examples for the UI (notes, trials with their criteria). Reads only the val split; gold labels are NOT exposed."""
from trialmatch.utils.common import load_config


class ExampleStore:
    MISSING_MSG = ("Validation examples are unavailable: data/processed/val.parquet is missing. "
                   "Run: python scripts/prepare_data.py")

    def __init__(self, rows: list[dict], error: str | None = None):
        """rows: dicts with annotation_id, patient_id, trial_id, trial_title, criterion_type, criterion_text, note."""
        self.rows, self.error = rows, error
        self.by_id = {r["annotation_id"]: r for r in rows}
        notes, trials = {}, {}
        for r in rows:
            notes.setdefault(r["patient_id"], {"id": r["patient_id"], "note": r["note"], "trials": []})
            if r["trial_id"] not in notes[r["patient_id"]]["trials"]:
                notes[r["patient_id"]]["trials"].append(r["trial_id"])
            t = trials.setdefault(r["trial_id"], {"id": r["trial_id"], "title": r.get("trial_title") or "",
                                                  "patient_id": r["patient_id"], "criteria": []})
            t["criteria"].append({"type": r["criterion_type"], "text": r["criterion_text"]})
        for n in notes.values():
            n["label"] = f"{n['id']} (annotated with {', '.join(n['trials'])})"
        for t in trials.values():
            t["label"] = f"{t['id']}: {t['title'][:70]} ({len(t['criteria'])} criteria)"
        self.notes, self.trials = list(notes.values()), list(trials.values())

    @classmethod
    def safe_from_val(cls) -> "ExampleStore":
        """from_val(), but never raises: an empty store carrying a clear message the UI shows instead."""
        try:
            return cls.from_val()
        except FileNotFoundError:
            return cls([], cls.MISSING_MSG)
        except Exception as e:
            return cls([], f"Validation examples could not be loaded ({type(e).__name__}). "
                           "If data/processed/ was not built yet, run: python scripts/prepare_data.py")

    @classmethod
    def from_val(cls) -> "ExampleStore":
        import pandas as pd

        from trialmatch.evaluation.val_data import load_gold
        cfg = load_config()
        c = cfg["columns"]
        df = load_gold("val", cfg)
        path = None
        try:  # trial_title is not in load_gold's column allow-list; read it separately from the same val file
            from trialmatch.utils.common import resolve
            path = resolve(cfg["paths"]["processed_dir"]) / "val.parquet"
            titles = pd.read_parquet(path, columns=[c["id"], "trial_title"]).set_index(c["id"])["trial_title"].to_dict()
        except Exception:
            titles = {}
        rows = [{"annotation_id": int(r[c["id"]]), "patient_id": r[c["patient"]], "trial_id": r[c["trial"]],
                 "trial_title": titles.get(int(r[c["id"]]), ""), "criterion_type": r[c["criterion_type"]],
                 "criterion_text": r[c["criterion_text"]], "note": r[c["note"]]} for _, r in df.iterrows()]
        return cls(rows)

    def public(self) -> dict:
        return {"error": self.error,
                "notes": [{k: n[k] for k in ("id", "label", "note")} for n in self.notes],
                "trials": [{k: t[k] for k in ("id", "label", "patient_id", "criteria")} for t in self.trials]}

    def prompts(self) -> list[dict]:
        return [{"id": r["annotation_id"],
                 "label": f"#{r['annotation_id']} {r['criterion_type']}: {str(r['criterion_text'])[:60]}"}
                for r in self.rows]
