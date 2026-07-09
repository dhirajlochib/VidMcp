"""Minimal human-in-the-loop review UI (stdlib HTTP server).

Serves project renders/previews and collects approve/reject + notes.
No FastAPI required — always available.
"""

from __future__ import annotations

import html
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import orjson

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.review")

_STATE: dict[str, Any] = {
    "server": None,
    "thread": None,
    "port": None,
    "workspace": None,
    "decisions": [],
}


def get_review_state() -> dict[str, Any]:
    return {
        "running": _STATE["server"] is not None,
        "port": _STATE["port"],
        "url": f"http://127.0.0.1:{_STATE['port']}/" if _STATE["port"] else None,
        "decisions": list(_STATE["decisions"][-50:]),
    }


def _list_projects(workspace: Path) -> list[dict[str, Any]]:
    items = []
    for child in sorted(Path(workspace).iterdir()):
        man = child / "manifest.json"
        if not man.exists():
            continue
        try:
            data = orjson.loads(man.read_bytes())
            renders = data.get("renders") or []
            items.append(
                {
                    "id": data.get("id", child.name),
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "renders": len(renders),
                    "last_render": renders[-1].get("output_path") if renders else None,
                }
            )
        except Exception:
            continue
    return items


def _page(workspace: Path) -> str:
    projects = _list_projects(workspace)
    rows = []
    for p in projects:
        rid = html.escape(str(p["id"]))
        name = html.escape(str(p.get("name") or ""))
        rows.append(
            f"<tr><td><a href='/project?id={rid}'>{name}</a></td>"
            f"<td><code>{rid[:8]}</code></td><td>{p.get('status')}</td>"
            f"<td>{p.get('renders')}</td></tr>"
        )
    decisions = "".join(
        f"<li><code>{html.escape(d.get('project_id','')[:8])}</code> "
        f"<b>{html.escape(d.get('decision',''))}</b> — {html.escape(d.get('note') or '')}</li>"
        for d in reversed(_STATE["decisions"][-20:])
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>VidMCP Review</title>
<style>
body {{ font-family: ui-sans-serif, system-ui; margin: 2rem; background:#0b0d12; color:#e8eaed; }}
a {{ color:#7dd3fc; }} table {{ border-collapse: collapse; width:100%; }}
td,th {{ border-bottom:1px solid #222; padding:8px; text-align:left; }}
.card {{ background:#141824; padding:1rem 1.25rem; border-radius:12px; margin:1rem 0; }}
button,input,textarea {{ font: inherit; }}
button {{ background:#2563eb; color:white; border:0; padding:8px 14px; border-radius:8px; cursor:pointer; margin-right:8px; }}
button.reject {{ background:#b91c1c; }}
textarea {{ width:100%; min-height:70px; background:#0b0d12; color:#e8eaed; border:1px solid #333; border-radius:8px; padding:8px; }}
</style></head><body>
<h1>VidMCP Review UI</h1>
<p>Human-in-the-loop approvals for renders. Workspace: <code>{html.escape(str(workspace))}</code></p>
<div class="card"><h2>Projects</h2>
<table><tr><th>Name</th><th>ID</th><th>Status</th><th>Renders</th></tr>
{''.join(rows) or '<tr><td colspan=4>No projects yet</td></tr>'}
</table></div>
<div class="card"><h2>Recent decisions</h2><ul>{decisions or '<li>None yet</li>'}</ul></div>
</body></html>"""


def _project_page(workspace: Path, project_id: str) -> str:
    root = Path(workspace) / project_id
    man_path = root / "manifest.json"
    if not man_path.exists():
        # try scan
        for child in Path(workspace).iterdir():
            mp = child / "manifest.json"
            if mp.exists():
                data = orjson.loads(mp.read_bytes())
                if data.get("id") == project_id or child.name == project_id:
                    root = child
                    man_path = mp
                    break
    if not man_path.exists():
        return f"<h1>Project not found: {html.escape(project_id)}</h1>"
    data = orjson.loads(man_path.read_bytes())
    renders = data.get("renders") or []
    previews = list((root / "previews").glob("*.jpg"))[:8] if (root / "previews").exists() else []
    render_links = []
    for r in renders[-5:]:
        rel = r.get("output_path")
        if not rel:
            continue
        render_links.append(f"<li><a href='/file?project={html.escape(root.name)}&path={html.escape(rel)}'>{html.escape(rel)}</a></li>")
    preview_imgs = "".join(
        f"<img src='/file?project={html.escape(root.name)}&path=previews/{html.escape(p.name)}' style='max-width:220px;margin:6px;border-radius:8px'/>"
        for p in previews
    )
    pid = html.escape(str(data.get("id")))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Review {html.escape(str(data.get('name')))}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui; margin: 2rem; background:#0b0d12; color:#e8eaed; }}
a {{ color:#7dd3fc; }} .card {{ background:#141824; padding:1rem 1.25rem; border-radius:12px; margin:1rem 0; }}
button {{ background:#2563eb; color:white; border:0; padding:8px 14px; border-radius:8px; cursor:pointer; margin-right:8px; }}
button.reject {{ background:#b91c1c; }}
textarea {{ width:100%; min-height:70px; background:#0b0d12; color:#e8eaed; border:1px solid #333; border-radius:8px; padding:8px; }}
</style></head><body>
<p><a href="/">← All projects</a></p>
<h1>{html.escape(str(data.get('name')))}</h1>
<p>Status: <b>{html.escape(str(data.get('status')))}</b> · ID <code>{pid}</code></p>
<div class="card"><h2>Previews</h2>{preview_imgs or '<p>No preview jpgs</p>'}</div>
<div class="card"><h2>Renders</h2><ul>{''.join(render_links) or '<li>None</li>'}</ul></div>
<div class="card"><h2>Decision</h2>
<form method="POST" action="/decide">
<input type="hidden" name="project_id" value="{pid}"/>
<textarea name="note" placeholder="Optional notes for the agent..."></textarea><br/><br/>
<button type="submit" name="decision" value="approve">Approve</button>
<button class="reject" type="submit" name="decision" value="reject">Reject</button>
<button type="submit" name="decision" value="revise">Request revise</button>
</form></div>
</body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    workspace: Path = Path(".")

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("review_http", msg=fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path in ("/", "/index"):
            body = _page(self.workspace).encode()
            self._ok(body, "text/html; charset=utf-8")
            return
        if u.path == "/project":
            qs = parse_qs(u.query)
            pid = (qs.get("id") or [""])[0]
            body = _project_page(self.workspace, pid).encode()
            self._ok(body, "text/html; charset=utf-8")
            return
        if u.path == "/api/decisions":
            self._ok(orjson.dumps(_STATE["decisions"]), "application/json")
            return
        if u.path == "/api/projects":
            self._ok(orjson.dumps(_list_projects(self.workspace)), "application/json")
            return
        if u.path == "/file":
            qs = parse_qs(u.query)
            proj = (qs.get("project") or [""])[0]
            rel = (qs.get("path") or [""])[0]
            path = (self.workspace / proj / rel).resolve()
            try:
                path.relative_to(self.workspace.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not path.exists() or not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            ctype = "application/octet-stream"
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                ctype = "image/jpeg"
            elif path.suffix.lower() == ".png":
                ctype = "image/png"
            elif path.suffix.lower() == ".mp4":
                ctype = "video/mp4"
            self._ok(data, ctype)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode()
        form = parse_qs(raw)
        if u.path == "/decide":
            decision = {
                "project_id": (form.get("project_id") or [""])[0],
                "decision": (form.get("decision") or [""])[0],
                "note": (form.get("note") or [""])[0],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _STATE["decisions"].append(decision)
            # persist
            try:
                p = Path(self.workspace) / ".vidmcp" / "review_decisions.jsonl"
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "ab") as f:
                    f.write(orjson.dumps(decision) + b"\n")
                # also write into project if found
                for child in Path(self.workspace).iterdir():
                    man = child / "manifest.json"
                    if not man.exists():
                        continue
                    data = orjson.loads(man.read_bytes())
                    if data.get("id") == decision["project_id"] or child.name == decision["project_id"]:
                        data.setdefault("reviews", []).append(decision)
                        man.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
                        break
            except Exception as e:  # noqa: BLE001
                log.warning("persist_decision_failed", error=str(e))
            body = f"""<!doctype html><html><body style="font-family:system-ui;background:#0b0d12;color:#eee;padding:2rem">
            <h1>Recorded: {html.escape(decision['decision'])}</h1>
            <p><a href="/project?id={html.escape(decision['project_id'])}">Back to project</a> · <a href="/">All projects</a></p>
            </body></html>""".encode()
            self._ok(body, "text/html; charset=utf-8")
            return
        self.send_error(404)

    def _ok(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_review_server(workspace: Path, port: int = 8765) -> dict[str, Any]:
    if _STATE["server"] is not None:
        return get_review_state() | {"ok": True, "message": "already running"}

    class Handler(ReviewHandler):
        pass

    Handler.workspace = Path(workspace)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    _STATE["server"] = server
    _STATE["port"] = port
    _STATE["workspace"] = str(workspace)

    def _run() -> None:
        log.info("review_ui_start", port=port)
        server.serve_forever()

    t = threading.Thread(target=_run, name="vidmcp-review-ui", daemon=True)
    t.start()
    _STATE["thread"] = t
    return {"ok": True, **get_review_state()}


def stop_review_server() -> dict[str, Any]:
    srv = _STATE.get("server")
    if srv:
        srv.shutdown()
        _STATE["server"] = None
        _STATE["port"] = None
    return {"ok": True, "running": False}
