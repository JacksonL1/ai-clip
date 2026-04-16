from pathlib import Path
from typing import Dict, Any
import subprocess
import asyncio
from pipeline.step1_outline import run_step1_outline
from pipeline.step2_timeline import run_step2_timeline
from pipeline.step3_scoring import run_step3_scoring
from pipeline.step4_title import run_step4_title
from pipeline.step5_clustering import run_step5_clustering
from pipeline.step6_video import run_step6_video


async def generate_subtitle_automatically(video_path: str, metadata_dir: Path) -> Path:
    """
    自动生成字幕文件

    Args:
        video_path: 视频文件路径
        metadata_dir: 元数据目录

    Returns:
        生成的SRT文件路径，如果失败返回None
    """
    try:
        print(f"开始为视频 {video_path} 自动生成字幕")

        video_file_path = Path(video_path)
        if not video_file_path.exists():
            print(f"视频文件不存在: {video_path}")
            return None

        metadata_dir.mkdir(parents=True, exist_ok=True)
        expected_srt_path = metadata_dir / f"{video_file_path.stem}.srt"
        whisper_cmd = [
            "whisper",
            str(video_file_path),
            "--model",
            "medium",
            "--output_format",
            "srt",
            "--output_dir",
            str(metadata_dir),
        ]

        print("使用命令行 whisper 自动生成字幕")
        # Avoid UnicodeDecodeError on Windows (gbk) by decoding manually.
        asr_result = subprocess.run(whisper_cmd, capture_output=True, text=False)
        stderr_text = (asr_result.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout_text = (asr_result.stdout or b"").decode("utf-8", errors="replace").strip()

        if asr_result.returncode != 0:
            print(f"whisper 执行失败: {stderr_text or stdout_text}")
            return None

        if expected_srt_path.exists():
            print(f"whisper 生成字幕成功: {expected_srt_path}")
            print(f"SUBTITLE", "AI字幕生成完成, 总进度40%, {expected_srt_path}")
            return expected_srt_path

        candidates = sorted(metadata_dir.glob(f"{video_file_path.stem}*.srt"))
        if candidates:
            print(f"whisper 生成字幕成功: {candidates[0]}")
            print("SUBTITLE", "AI字幕生成完成, 总进度40%")
            return candidates[0]

        print("whisper 执行成功但未找到 SRT 输出文件")
        return None

    except Exception as e:
        print(f"自动生成字幕过程中发生错误: {e}")
        return None


class SimplePipelineAdapter:
    """简化的流水线适配器，使用固定阶段进度系统"""

    def __init__(self, project_id: str, task_id: str):
        self.project_id = project_id
        self.task_id = task_id

    async def process_project_sync(self, input_video_path: str, input_srt_path: str) -> Dict[str, Any]:
        """
        同步处理项目 - 使用简化的进度系统

        Args:
            input_video_path: 输入视频路径
            input_srt_path: 输入SRT路径

        Returns:
            处理结果
        """
        print(f"开始处理项目: {self.project_id}")

        try:
            # 创建必要的目录结构 - 使用正确的路径
            from core.path_utils import get_project_directory
            project_dir = get_project_directory(self.project_id)
            metadata_dir = project_dir / "metadata"
            output_dir = project_dir / "output"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(metadata_dir)
            # 项目内专属输出子目录
            clips_output_dir = output_dir / "clips"
            collections_output_dir = output_dir / "collections"
            clips_output_dir.mkdir(parents=True, exist_ok=True)
            collections_output_dir.mkdir(parents=True, exist_ok=True)

            # 阶段1: 素材准备
            print(self.project_id, "INGEST", "素材准备完成")

            # 阶段2: 字幕处理
            print(self.project_id, "SUBTITLE", "开始字幕处理")

            # Step 1: 大纲提取
            print("执行Step 1: 大纲提取")
            if input_srt_path and Path(input_srt_path).exists():
                print(f"使用现有SRT文件: {input_srt_path}")
                outlines = run_step1_outline(Path(input_srt_path), metadata_dir=metadata_dir)
            else:
                print("没有SRT文件，尝试自动生成字幕")
                # 尝试自动生成字幕
                srt_path = await generate_subtitle_automatically(input_video_path, metadata_dir)
                if srt_path and srt_path.exists():
                    print(f"自动生成字幕成功: {srt_path}")
                    outlines = run_step1_outline(srt_path, metadata_dir=metadata_dir)
                else:
                    print("自动生成字幕失败，创建空大纲")
                    # 创建一个空的大纲文件
                    outlines = []
                    outline_file = metadata_dir / "step1_outline.json"
                    import json
                    with open(outline_file, 'w', encoding='utf-8') as f:
                        json.dump(outlines, f, ensure_ascii=False, indent=2)
            print(self.project_id, "SUBTITLE", "字幕处理完成， 进度：50")

            # 阶段3: 内容分析
            print(self.project_id, "ANALYZE", "开始内容分析")

            # Step 2: 时间线提取
            print("执行Step 2: 时间线提取")
            if outlines:  # 只有当有大纲时才执行后续步骤
                timeline_data = run_step2_timeline(
                    metadata_dir / "step1_outline.json",
                    metadata_dir=metadata_dir
                )
                print(self.project_id, "ANALYZE", "时间线提取完成， 进度50")

                # Step 3: 内容评分
                print("执行Step 3: 内容评分")
                scored_clips = run_step3_scoring(
                    metadata_dir / "step2_timeline.json",
                    metadata_dir=metadata_dir
                )
                print(self.project_id, "ANALYZE", "内容分析完成， 进度100")
            else:
                print("没有大纲数据，跳过时间线提取和内容评分")
                # 创建空的时间线和评分文件
                timeline_file = metadata_dir / "step2_timeline.json"
                scored_file = metadata_dir / "step3_high_score_clips.json"
                import json
                with open(timeline_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                with open(scored_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                # 初始化空变量
                timeline_data = []
                scored_clips = []
                print(self.project_id, "ANALYZE", "内容分析完成， 进度100")

            # 阶段4: 片段定位
            print(self.project_id, "HIGHLIGHT", "开始片段定位")

            # Step 4: 标题生成
            print("执行Step 4: 标题生成")
            if outlines:  # 只有当有大纲时才执行后续步骤
                titled_clips = run_step4_title(
                    metadata_dir / "step3_high_score_clips.json",
                    metadata_dir=str(metadata_dir)
                )
                print(self.project_id, "HIGHLIGHT", "标题生成完成，进度40")

                # Step 5: 主题聚类
                print("执行Step 5: 主题聚类")
                collections = run_step5_clustering(
                    metadata_dir / "step4_titles.json",
                    metadata_dir=str(metadata_dir)
                )
                print(self.project_id, "HIGHLIGHT", "片段定位完成, 进度100")

                # 阶段5: 视频导出
                print(self.project_id, "EXPORT", "开始视频导出")

                # Step 6: 视频切割
                print("执行Step 6: 视频切割")
                video_result = run_step6_video(
                    metadata_dir / "step4_titles.json",
                    metadata_dir / "step5_collections.json",
                    input_video_path,
                    output_dir=output_dir,
                    clips_dir=str(clips_output_dir),
                    collections_dir=str(collections_output_dir),
                    metadata_dir=str(metadata_dir)
                )
            else:
                print("没有大纲数据，跳过标题生成、主题聚类和视频切割")
                # 创建空的标题和合集文件
                titles_file = metadata_dir / "step4_titles.json"
                collections_file = metadata_dir / "step5_collections.json"
                import json
                with open(titles_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                with open(collections_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                # 初始化空变量
                titled_clips = []
                collections = []
                print(self.project_id, "HIGHLIGHT", "片段定位完成，进度100")
                print(self.project_id, "EXPORT", "开始视频导出")
                video_result = {"status": "skipped", "message": "没有内容可处理"}
            print(self.project_id, "EXPORT", "视频导出完成，进度100")

            # 阶段6: 处理完成
            print(self.project_id, "DONE", "处理完成")

            print(f"项目处理完成: {self.project_id}")
            return {
                "status": "succeeded",
                "project_id": self.project_id,
                "task_id": self.task_id,
                "result": {
                    "outlines": outlines,
                    "timeline": timeline_data,
                    "scored_clips": scored_clips,
                    "titled_clips": titled_clips,
                    "collections": collections,
                    "video_result": video_result
                }
            }

        except Exception as e:
            error_msg = f"流水线处理失败: {str(e)}"
            print(error_msg)

            # 发送失败状态
            print(self.project_id, "DONE", f"处理失败: {error_msg}")

            return {
                "status": "failed",
                "project_id": self.project_id,
                "task_id": self.task_id,
                "error": error_msg
            }


project_id = "tennis1"
task_id = "001"
input_video_path = r"F:\Sports\pingpong\video\B5C8EEF0-3D05-11EF-B70C-F558F0C0988B.mp4"
input_srt_path = r"D:\project\aiclip\data\projects\sports1\metadata\B5C8EEF0-3D05-11EF-B70C-F558F0C0988B.srt"
project_sync = SimplePipelineAdapter(project_id, task_id)
result = asyncio.run(project_sync.process_project_sync(input_video_path, input_srt_path))
