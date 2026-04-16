from utils.llm_client import LLMClient
from pathlib import Path
from core.shared_config import METADATA_DIR


class DynamicSceneSegmenter:
    def __init__(self, metadata_dir: Path = None):
        self.llm_client = LLMClient()
        # 使用传入的metadata_dir或默认值
        if metadata_dir is None:
            metadata_dir = METADATA_DIR
        self.metadata_dir = metadata_dir

        self.user_prefs = {
            "granularity": "detailed",
            "style": "professional"
        }

    def _get_text_from_segments(self, segments):
        """将 Whisper 段落转为带时间戳的字符串"""
        return "\n".join([f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" for s in segments])

    def _summarize_context(self, context_text, llm_handler):
        """当上下文过长时，调用 LLM 进行摘要压缩"""
        summary_prompt = f"请简要总结以下视频片段的主题，以便为后续场景切分提供背景信息：\n{context_text}"
        return llm_handler(summary_prompt)

    def process_video(self, asr_file, llm_handler, window_minutes=8):
        """
        llm_handler: 一个函数，接收 prompt 并返回 LLM 的回复字符串
        """

        window_seconds = window_minutes * 60
        full_results = []
        last_context = ""  # 存储上一段的衔接信息

        # 按时间跨度进行切分
        start_idx = 0
        while start_idx < len(segments):
            current_window_segments = []
            current_time_total = 0
            end_idx = start_idx

            # 提取 5-10 分钟的片段
            while end_idx < len(segments) and current_time_total < window_seconds:
                current_window_segments.append(segments[end_idx])
                current_time_total = segments[end_idx]['end'] - segments[start_idx]['start']
                end_idx += 1

            current_text = self._get_text_from_segments(current_window_segments)

            # 构建动态 Prompt
            prompt = self._build_final_prompt(last_context, current_text)

            # 调用用户自定义的 LLM API
            response = llm_handler(prompt)
            full_results.append(response)

            # --- 动态上下文处理 ---
            # 假设我们将当前段落的最后 1 分钟作为下一段的“剩余部分”
            overlap_segments = current_window_segments[-5:]  # 取最后5句作为重叠衔接
            overlap_text = self._get_text_from_segments(overlap_segments)

            if len(overlap_text) > 1000:  # 如果衔接部分字数过长（阈值可调）
                last_context = "上文摘要: " + self._summarize_context(overlap_text, llm_handler)
            else:
                last_context = "上文衔接: " + overlap_text

            start_idx = end_idx  # 移动到下一窗口

        return full_results

    def _build_final_prompt(self, context, current_text):
        return f"""
### 角色
你是一个视频内容分析专家。

### 上下文背景 (Context)
{context}

### 任务
请根据以下 ASR 文本进行场景切分。
用户习惯偏好：{self.user_prefs['granularity']}

### 当前待处理文本
{current_text}

### 输出要求
请识别出场景转换的时间点，并给出标题和简要描述。
"""


# --- 使用示例 ---

# 1. 定义用户自己的 LLM 调用逻辑 (例如使用 OpenAI)
def my_custom_llm(prompt):
    # 这里可以是任何 LLM SDK，如 openai.ChatCompletion.create(...)
    print(f"\n[发送 Prompt 到 LLM，长度: {len(prompt)}...]")
    return "LLM 返回的场景切分数据..."


# 2. 运行
segmenter = DynamicSceneSegmenter()

results = segmenter.process_video("meeting_record.mp4", my_custom_llm)
