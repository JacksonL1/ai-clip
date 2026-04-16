"""简单前端服务：展示流水线步骤并在线编辑 Prompt。"""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROMPT_ROOT = REPO_ROOT / "prompt"

PROMPT_NAME_TO_FILE = {
    "大纲": "大纲.txt",
    "时间点": "时间点.txt",
    "推荐理由": "推荐理由.txt",
    "标题生成": "标题生成.txt",
    "主题聚类": "主题聚类.txt",
}

STEP_DEFINITIONS = [
    {"id": "step1", "name": "Step 1 大纲提取", "desc": "从字幕生成结构化大纲"},
    {"id": "step2", "name": "Step 2 时间点提取", "desc": "从大纲定位时间段"},
    {"id": "step3", "name": "Step 3 推荐理由", "desc": "为候选片段评分并给出推荐理由"},
    {"id": "step4", "name": "Step 4 标题生成", "desc": "为高分片段生成标题"},
    {"id": "step5", "name": "Step 5 主题聚类", "desc": "聚合生成合集"},
    {"id": "step6", "name": "Step 6 视频导出", "desc": "输出切片与合集视频"},
]


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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


def list_prompt_categories() -> List[str]:
    categories = ["default"]
    for item in PROMPT_ROOT.iterdir():
        if item.is_dir():
            categories.append(item.name)
    return sorted(categories)


def get_prompt_dir(category: str) -> Path:
    return PROMPT_ROOT if category == "default" else PROMPT_ROOT / category


def load_prompts(category: str) -> Dict[str, str]:
    prompt_dir = get_prompt_dir(category)
    prompts: Dict[str, str] = {}
    for prompt_name, file_name in PROMPT_NAME_TO_FILE.items():
        prompts[prompt_name] = _safe_read(prompt_dir / file_name)
    return prompts


def save_prompt(category: str, prompt_name: str, content: str) -> Path:
    if prompt_name not in PROMPT_NAME_TO_FILE:
        raise ValueError(f"不支持的 Prompt 类型: {prompt_name}")
    prompt_dir = get_prompt_dir(category)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / PROMPT_NAME_TO_FILE[prompt_name]
    path.write_text(content, encoding="utf-8")
    return path


def run_single_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    from pipeline.step1_outline import run_step1_outline
    from pipeline.step2_timeline import run_step2_timeline
    from pipeline.step3_scoring import run_step3_scoring
    from pipeline.step4_title import run_step4_title
    from pipeline.step5_clustering import run_step5_clustering
    from pipeline.step6_video import run_step6_video

    step_id = payload.get("step_id")
    metadata_dir = Path(payload.get("metadata_dir", "data/output/metadata"))
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if step_id == "step1":
        srt_path = Path(payload["srt_path"])
        outlines = run_step1_outline(srt_path, metadata_dir=metadata_dir)
        return {"count": len(outlines), "output": str(metadata_dir / "step1_outline.json")}

    if step_id == "step2":
        outline_path = Path(payload.get("outline_path") or metadata_dir / "step1_outline.json")
        timeline = run_step2_timeline(outline_path, metadata_dir=metadata_dir)
        return {"count": len(timeline), "output": str(metadata_dir / "step2_timeline.json")}

    if step_id == "step3":
        timeline_path = Path(payload.get("timeline_path") or metadata_dir / "step2_timeline.json")
        scored = run_step3_scoring(timeline_path, metadata_dir=metadata_dir)
        return {"count": len(scored), "output": str(metadata_dir / "step3_high_score_clips.json")}

    if step_id == "step4":
        score_path = Path(payload.get("score_path") or metadata_dir / "step3_high_score_clips.json")
        titled = run_step4_title(score_path, metadata_dir=str(metadata_dir))
        return {"count": len(titled), "output": str(metadata_dir / "step4_titles.json")}

    if step_id == "step5":
        title_path = Path(payload.get("title_path") or metadata_dir / "step4_titles.json")
        clustered = run_step5_clustering(title_path, metadata_dir=str(metadata_dir))
        return {"count": len(clustered), "output": str(metadata_dir / "step5_collections.json")}

    if step_id == "step6":
        titles_path = Path(payload.get("title_path") or metadata_dir / "step4_titles.json")
        collections_path = Path(payload.get("collections_path") or metadata_dir / "step5_collections.json")
        video_path = payload["video_path"]
        output_dir = Path(payload.get("output_dir", "data/output"))
        clips_dir = output_dir / "clips"
        collections_dir = output_dir / "collections"
        result = run_step6_video(
            titles_path,
            collections_path,
            video_path,
            output_dir=output_dir,
            clips_dir=str(clips_dir),
            collections_dir=str(collections_dir),
            metadata_dir=str(metadata_dir),
        )
        return {"output": str(output_dir), "result": result}

    raise ValueError(f"未知步骤: {step_id}")


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            html_path = BASE_DIR / "static" / "index.html"
            body = html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/prompts":
            query = parse_qs(parsed.query)
            category = query.get("category", ["default"])[0]
            _json_response(
                self,
                {
                    "categories": list_prompt_categories(),
                    "active_category": category,
                    "prompts": load_prompts(category),
                },
            )
            return

        if parsed.path == "/api/steps":
            _json_response(self, {"steps": STEP_DEFINITIONS})
            return

        _json_response(self, {"error": "Not Found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        try:
            payload = _parse_json_body(self)
        except json.JSONDecodeError:
            _json_response(self, {"error": "JSON 格式错误"}, status=400)
            return

        if parsed.path == "/api/prompts/save":
            try:
                category = payload.get("category", "default")
                prompt_name = payload["prompt_name"]
                content = payload.get("content", "")
                path = save_prompt(category, prompt_name, content)
                _json_response(self, {"ok": True, "path": str(path)})
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/run-step":
            try:
                result = run_single_step(payload)
                _json_response(self, {"ok": True, "result": result})
            except Exception as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, status=400)
            return

        _json_response(self, {"error": "Not Found"}, status=404)


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"UI server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
