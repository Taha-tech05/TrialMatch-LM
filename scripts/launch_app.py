"""CLI: launch the browser app.

  python scripts/launch_app.py                       # auto: real model if a source exists, else demo
  python scripts/launch_app.py --mode demo           # cached validation predictions only (no torch needed)
  python scripts/launch_app.py --mode real --merged-dir PATH_TO_MERGED_FP16
  python scripts/launch_app.py --share               # also open a public link IF cloudflared or ngrok is installed

Then open http://localhost:7860 . Binds to 127.0.0.1 unless --host is given.
"""
import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SHARE_HELP = """\
--share needs a tunnel tool that is already installed; none was found. Options:
  * Colab:   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
             then re-run with --share (the script looks for ./cloudflared as well as PATH).
             Or in a notebook cell:  from google.colab import output; output.serve_kernel_port_as_iframe(7860)
  * ngrok:   install ngrok, run `ngrok config add-authtoken <token>`, then re-run with --share.
The app itself is running locally either way."""


def start_share(port: int) -> None:
    exe = shutil.which("cloudflared") or (str(Path("cloudflared").resolve()) if Path("cloudflared").exists() else None)
    if exe:
        p = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{port}"], stderr=subprocess.PIPE, text=True)

        def watch():
            for line in p.stderr:
                m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
                if m:
                    print(f"\n  Public share link: {m.group(0)}\n", flush=True)
                    return
        threading.Thread(target=watch, daemon=True).start()
        return
    if shutil.which("ngrok"):
        subprocess.Popen(["ngrok", "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def poll():
            import json
            import urllib.request
            for _ in range(30):
                time.sleep(1)
                try:
                    d = json.load(urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2))
                    print(f"\n  Public share link: {d['tunnels'][0]['public_url']}\n", flush=True)
                    return
                except Exception:
                    pass
            print("ngrok started but no tunnel URL was found; check `ngrok config add-authtoken`.", flush=True)
        threading.Thread(target=poll, daemon=True).start()
        return
    print(SHARE_HELP)


def main() -> int:
    from trialmatch.app.config import load_app_config
    cfg = load_app_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["real", "demo", "auto"], default="auto")
    ap.add_argument("--merged-dir", help="saved merged fp16 model directory (overrides config/env)")
    ap.add_argument("--port", type=int, default=cfg["server"]["port"])
    ap.add_argument("--host", default=cfg["server"]["host"])
    ap.add_argument("--share", action="store_true", help="open a public link via an installed cloudflared/ngrok")
    a = ap.parse_args()
    try:
        import uvicorn
        from trialmatch.app.runtime import Runtime
        from trialmatch.app.server import create_app
    except ImportError as e:
        print(f"Missing dependency: {e}. Install with: pip install -r requirements-app.txt")
        return 1
    rt = Runtime(cfg, a.mode, a.merged_dir)
    rt.start()
    app = create_app(rt)
    print(f"\n  TrialMatch screening app ({a.mode} mode requested)\n  Open http://localhost:{a.port}\n", flush=True)
    if a.share:
        start_share(a.port)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
