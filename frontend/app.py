"""前端演示服务：一键拆条 + 步骤输出展示 + Prompt 文件编辑。"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROMPT_ROOT = REPO_ROOT / "prompt"
RUNS_ROOT = REPO_ROOT / "data" / "demo_runs"
RUN_INDEX_FILE = RUNS_ROOT / "latest_run.json"

STEP_LABELS = [
    ("step1", "大纲提取"),
    ("step2", "时间点提取"),
    ("step3", "推荐理由评分"),
    ("step4", "标题生成"),
    ("step5", "主题聚类"),
    ("step6", "视频导出"),
]


def _json_response(handler: BaseHTTPRequestHandler, data: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _safe_json(path: Path) -> Any:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_prompt_categories() -> List[str]:
    categories = ["default"]
    for item in PROMPT_ROOT.iterdir():
        if item.is_dir():
            categories.append(item.name)
    return sorted(categories)


def get_prompt_dir(category: str) -> Path:
    return PROMPT_ROOT if category == "default" else PROMPT_ROOT / category


def list_prompt_files(category: str) -> List[str]:
    prompt_dir = get_prompt_dir(category)
    if not prompt_dir.exists():
        return []
    return sorted([p.name for p in prompt_dir.glob("*.txt")])


def read_prompt_file(category: str, filename: str) -> str:
    path = get_prompt_dir(category) / filename
    return _safe_read(path)


def save_prompt_file(category: str, filename: str, content: str) -> Path:
    prompt_dir = get_prompt_dir(category)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".txt"):
        raise ValueError("仅允许编辑 .txt Prompt 文件")
    path = prompt_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def _build_run_paths(run_id: str) -> Dict[str, Path]:
    run_base = RUNS_ROOT / run_id
    output_dir = run_base / "output"
    metadata_dir = output_dir / "metadata"
    clips_dir = output_dir / "clips"
    collections_dir = output_dir / "collections"
    thumbs_dir = output_dir / "thumbnails"
    for d in [run_base, output_dir, metadata_dir, clips_dir, collections_dir, thumbs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "run_base": run_base,
        "output_dir": output_dir,
        "metadata_dir": metadata_dir,
        "clips_dir": clips_dir,
        "collections_dir": collections_dir,
        "thumbs_dir": thumbs_dir,
    }


def _extract_thumbnail(video_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss",
        "1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return result.returncode == 0 and output_path.exists()


def _generate_subtitle(video_path: Path, metadata_dir: Path) -> Path | None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    expected_srt_path = metadata_dir / f"{video_path.stem}.srt"
    whisper_cmd = [
        "whisper",
        str(video_path),
        "--model",
        "medium",
        "--output_format",
        "srt",
        "--output_dir",
        str(metadata_dir),
    ]
    result = subprocess.run(whisper_cmd, capture_output=True, text=False)
    if result.returncode != 0:
        return None
    if expected_srt_path.exists():
        return expected_srt_path
    candidates = sorted(metadata_dir.glob(f"{video_path.stem}*.srt"))
    return candidates[0] if candidates else None


def _preview_step_output(step_file: Path, limit: int = 2) -> Any:
    data = _safe_json(step_file)
    if isinstance(data, list):
        return data[:limit]
    return data


def run_pipeline(video_path: str, srt_path: str | None = None) -> Dict[str, Any]:
    from pipeline.step1_outline import run_step1_outline
    from pipeline.step2_timeline import run_step2_timeline
    from pipeline.step3_scoring import run_step3_scoring
    from pipeline.step4_title import run_step4_title
    from pipeline.step5_clustering import run_step5_clustering
    from pipeline.step6_video import run_step6_video

    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"视频文件不存在: {input_video}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    paths = _build_run_paths(run_id)
    metadata_dir = paths["metadata_dir"]

    real_srt_path: Path | None
    if srt_path:
        real_srt_path = Path(srt_path)
        if not real_srt_path.exists():
            raise FileNotFoundError(f"字幕文件不存在: {real_srt_path}")
    else:
        real_srt_path = _generate_subtitle(input_video, metadata_dir)

    steps: List[Dict[str, Any]] = []

    start = time.time()
    if real_srt_path is None:
        raise RuntimeError("未提供字幕且自动生成字幕失败，请填写字幕路径或安装 whisper。")

    outlines = run_step1_outline(real_srt_path, metadata_dir=metadata_dir)
    steps.append({
        "id": "step1",
        "name": "大纲提取",
        "status": "done",
        "count": len(outlines),
        "preview": _preview_step_output(metadata_dir / "step1_outline.json"),
    })

    timeline = run_step2_timeline(metadata_dir / "step1_outline.json", metadata_dir=metadata_dir)
    steps.append({
        "id": "step2",
        "name": "时间点提取",
        "status": "done",
        "count": len(timeline),
        "preview": _preview_step_output(metadata_dir / "step2_timeline.json"),
    })

    scored = run_step3_scoring(metadata_dir / "step2_timeline.json", metadata_dir=metadata_dir)
    steps.append({
        "id": "step3",
        "name": "推荐理由评分",
        "status": "done",
        "count": len(scored),
        "preview": _preview_step_output(metadata_dir / "step3_high_score_clips.json"),
    })

    titled = run_step4_title(metadata_dir / "step3_high_score_clips.json", metadata_dir=str(metadata_dir))
    steps.append({
        "id": "step4",
        "name": "标题生成",
        "status": "done",
        "count": len(titled),
        "preview": _preview_step_output(metadata_dir / "step4_titles.json"),
    })

    clustered = run_step5_clustering(metadata_dir / "step4_titles.json", metadata_dir=str(metadata_dir))
    steps.append({
        "id": "step5",
        "name": "主题聚类",
        "status": "done",
        "count": len(clustered),
        "preview": _preview_step_output(metadata_dir / "step5_collections.json"),
    })

    video_result = run_step6_video(
        metadata_dir / "step4_titles.json",
        metadata_dir / "step5_collections.json",
        input_video,
        output_dir=paths["output_dir"],
        clips_dir=str(paths["clips_dir"]),
        collections_dir=str(paths["collections_dir"]),
        metadata_dir=str(metadata_dir),
    )
    steps.append({
        "id": "step6",
        "name": "视频导出",
        "status": "done",
        "count": video_result.get("clips_generated", 0),
        "preview": video_result,
    })

    titled_items = _safe_json(metadata_dir / "step4_titles.json")
    title_map = {str(item.get("id")): item.get("generated_title", f"片段_{item.get('id')}") for item in titled_items}

    clips: List[Dict[str, str]] = []
    for clip_path_str in video_result.get("clip_paths", []):
        clip_path = Path(clip_path_str)
        clip_id = clip_path.stem.split("_", 1)[0]
        thumb_path = paths["thumbs_dir"] / f"{clip_path.stem}.jpg"
        if _extract_thumbnail(clip_path, thumb_path):
            thumbnail_url = f"/api/media?path={thumb_path.resolve()}"
        else:
            thumbnail_url = ""
        clips.append({
            "clip_id": clip_id,
            "title": title_map.get(clip_id, clip_path.stem),
            "video_url": f"/api/media?path={clip_path.resolve()}",
            "thumbnail_url": thumbnail_url,
        })

    summary = {
        "run_id": run_id,
        "video_path": str(input_video),
        "srt_path": str(real_srt_path),
        "duration_seconds": round(time.time() - start, 2),
        "steps": steps,
        "clips": clips,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_path = paths["run_base"] / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    RUN_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_INDEX_FILE.write_text(json.dumps({"latest_run_id": run_id}, ensure_ascii=False), encoding="utf-8")
    return summary


def _assert_allowed_media(path_str: str) -> Path:
    raw = Path(path_str)
    resolved = raw.resolve()
    allowed_roots = [RUNS_ROOT.resolve()]
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        raise PermissionError("不允许访问该文件")
    if not resolved.exists():
        raise FileNotFoundError("文件不存在")
    return resolved


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self._serve_html("index.html")
            return
        if parsed.path == "/prompts":
            self._serve_html("prompts.html")
            return
        if parsed.path == "/results":
            self._serve_html("results.html")
            return

        if parsed.path.startswith("/static/"):
            self._serve_static(parsed.path)
            return

        if parsed.path == "/api/prompt-files":
            category = parse_qs(parsed.query).get("category", ["default"])[0]
            _json_response(self, {
                "categories": list_prompt_categories(),
                "active_category": category,
                "files": list_prompt_files(category),
            })
            return

        if parsed.path == "/api/prompt-file":
            query = parse_qs(parsed.query)
            category = query.get("category", ["default"])[0]
            filename = query.get("file", [""])[0]
            _json_response(self, {
                "category": category,
                "file": filename,
                "content": read_prompt_file(category, filename),
            })
            return

        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            run_id = query.get("id", [""])[0]
            if not run_id and RUN_INDEX_FILE.exists():
                run_id = json.loads(RUN_INDEX_FILE.read_text(encoding="utf-8")).get("latest_run_id", "")
            summary_path = RUNS_ROOT / run_id / "run_summary.json"
            if not summary_path.exists():
                _json_response(self, {"error": "未找到运行结果"}, status=404)
                return
            _json_response(self, _safe_json(summary_path))
            return

        if parsed.path == "/api/media":
            path_str = parse_qs(parsed.query).get("path", [""])[0]
            try:
                media_path = _assert_allowed_media(path_str)
                data = media_path.read_bytes()
                ctype = "image/jpeg" if media_path.suffix.lower() in {".jpg", ".jpeg"} else "video/mp4"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=403)
            return

        _json_response(self, {"error": "Not Found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = _parse_json_body(self)
        except json.JSONDecodeError:
            _json_response(self, {"error": "JSON 格式错误"}, status=400)
            return

        if parsed.path == "/api/prompt-file/save":
            try:
                category = payload.get("category", "default")
                filename = payload.get("file", "")
                content = payload.get("content", "")
                path = save_prompt_file(category, filename, content)
                _json_response(self, {"ok": True, "path": str(path)})
            except Exception as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
            return

        if parsed.path == "/api/run-pipeline":
            try:
                video_path = payload.get("video_path", "").strip()
                srt_path = payload.get("srt_path", "").strip() or None
                if not video_path:
                    raise ValueError("视频地址为必填项")
                summary = run_pipeline(video_path=video_path, srt_path=srt_path)
                _json_response(self, {"ok": True, "run_id": summary["run_id"], "summary": summary})
            except Exception as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
            return

        _json_response(self, {"error": "Not Found"}, status=404)

    def _serve_html(self, name: str) -> None:
        html_path = BASE_DIR / "static" / name
        body = html_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        static_path = BASE_DIR / path.lstrip("/")
        if not static_path.exists():
            _json_response(self, {"error": "Not Found"}, status=404)
            return
        body = static_path.read_bytes()
        ctype = "text/css" if static_path.suffix == ".css" else "application/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"UI server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
