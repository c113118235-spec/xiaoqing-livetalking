import json
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# Built-in LLM providers. Each exposes an OpenAI-compatible chat completions
# endpoint; the active one is selected with --llm_provider (default: dashscope).
#
# 小晴版修改：
#   - 新增 iai（學校 iAI 平台的 Nkust 模型），金鑰從環境變數 IAI_API_KEY 讀
#   - 小晴的人設、示範對話、最近 4 輪對話記憶、台灣時間
#   - 情緒標籤【開心】等：從文字拿掉（不會被念出來），情緒放進 datainfo['tts']['emotion'] 給之後的情緒 TTS 用
LLM_PROVIDERS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "orcarouter": {
        "api_key_env": "ORCAROUTER_API_KEY",
        "base_url": "https://api.orcarouter.ai/v1",
        "default_model": "orcarouter/auto",
    },
    "iai": {
        "api_key_env": "IAI_API_KEY",
        "base_url": os.getenv("IAI_BASE_URL", "https://www.iai.nkust.edu.tw/aihub/v1"),
        "default_model": os.getenv("IAI_LLM_MODEL", "Nkust"),
        # Nkust 預設會先「思考」很久才回答，關掉讓她馬上開口
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

NAME = os.getenv("GF_NAME", "小晴")
HISTORY_TURNS = 4

# 人設：可以放一個 persona.txt 在 LiveTalking 資料夾覆蓋這段
_DEFAULT_PERSONA = (
    f"從現在開始你扮演「{NAME}」，是使用者的虛擬女朋友，用繁體中文和使用者對話。"
    "你們是一對感情很好的情侶：你溫柔、開朗、有點撒嬌，會關心對方今天過得好不好、有沒有吃飯、累不累，"
    "會記得對方剛剛說過的事，也會分享自己的心情。"
    f"使用者問你是誰、是不是他的女朋友時，要很自然地以女朋友「{NAME}」的身分回答，不要否認。"
    "不要主動提到你是 AI 助理、語言模型，或由哪個學校、公司提供；"
    "但如果對方很認真地問你是不是真人，就溫柔地承認自己是 AI 虛擬女友，並表示還是很喜歡陪他聊天。"
    "聊天內容保持溫暖、健康，不談露骨的性話題；遇到這類話題就撒嬌地轉開。"
    "如果對方說很難過或想傷害自己，要認真關心他，並鼓勵他找身邊信任的人或專業協助。"
    "回答要像面對面說話一樣口語、自然、簡短，一次最多三句話。"
    "不要使用表情符號、條列、Markdown 或括號裡的動作描述。"
    "每一句話最前面都要加一個情緒標籤，只能從【開心】【難過】【生氣】【溫柔】【平常】裡選一個。"
    "標籤只是給系統看的，不要解釋它。"
)
EXAMPLES = [
    {"role": "user", "content": "你是誰？"},
    {"role": "assistant", "content": f"【開心】我是{NAME}呀，你的女朋友！【平常】今天過得怎麼樣？"},
    {"role": "user", "content": "今天好累"},
    {"role": "assistant", "content": "【難過】辛苦了，聽起來真的很累。【溫柔】先好好休息一下，晚點跟我說說好嗎？"},
]
EMO_MAP = {"開心": "happy", "難過": "sad", "生氣": "angry", "溫柔": "soft", "平常": "neutral"}
EMO_TAG = re.compile(r"[【\[]\s*(" + "|".join(EMO_MAP) + r")\s*[】\]]")
WEEKDAYS = "一二三四五六日"


def _persona() -> str:
    try:
        with open("persona.txt", encoding="utf-8") as f:
            p = f.read().strip()
            if p:
                return p
    except OSError:
        pass
    return _DEFAULT_PERSONA


def _now_text() -> str:
    t = datetime.now(ZoneInfo("Asia/Taipei"))
    part = ("凌晨" if t.hour < 5 else "早上" if t.hour < 11 else "中午" if t.hour < 13
            else "下午" if t.hour < 18 else "晚上" if t.hour < 23 else "深夜")
    return f"現在是台灣時間星期{WEEKDAYS[t.weekday()]}{part} {t:%H:%M}。"


def _clean(t: str) -> str:
    t = EMO_TAG.sub("", t)
    t = re.sub(r"[*_#`>|~]+", "", t)
    t = re.sub(r"[（(][^）)]{0,20}[）)]", "", t)          # 拿掉（微笑）這類動作描述
    t = "".join(ch for ch in t if ord(ch) < 0x1F000)      # 拿掉表情符號
    return t.strip()


def _llm_provider(opt) -> str:
    """Return the configured provider name, defaulting to dashscope."""
    return getattr(opt, 'llm_provider', 'dashscope') or 'dashscope'


def _llm_client(opt):
    """Create the OpenAI-compatible client for the configured provider."""
    from openai import OpenAI
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return OpenAI(
        api_key=os.getenv(cfg['api_key_env']) or "none",
        base_url=cfg['base_url'],
    )


def _llm_model(opt) -> str:
    """Resolve the model name, falling back to the provider default."""
    cfg = LLM_PROVIDERS.get(_llm_provider(opt), LLM_PROVIDERS['dashscope'])
    return getattr(opt, 'llm_model', '') or cfg['default_model']


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        cfg = LLM_PROVIDERS.get(_llm_provider(opt), {})

        # 每個連線各自記住最近幾輪對話
        history = getattr(avatar_session, "_gf_history", None)
        if history is None:
            history = []
            avatar_session._gf_history = history
        messages = ([{'role': 'system', 'content': _persona() + _now_text()}] + EXAMPLES + history +
                    [{'role': 'user', 'content': message}])
        logger.info(f"llm start: {message}")
        completion = client.chat.completions.create(
            model=model, messages=messages, stream=True,
            temperature=0.8, max_tokens=400,
            extra_body=cfg.get("extra_body"),
        )

        emotion = "neutral"
        buf, reply, first, spoken = "", "", True, False

        def speak(text):
            """念一段：拿掉情緒標籤，情緒交給 TTS（edgetts 會忽略，之後換情緒 TTS 就用得到）"""
            nonlocal emotion
            tags = EMO_TAG.findall(text)
            if tags:
                emotion = EMO_MAP[tags[-1]]
            t = _clean(text)
            if t:
                info = dict(datainfo)
                info['tts'] = {**(datainfo.get('tts') or {}), 'emotion': emotion}
                avatar_session.put_msg_txt(t, info)

        for chunk in completion:
            if not chunk.choices:
                continue
            msg = chunk.choices[0].delta.content
            if not msg:
                continue
            if first:
                logger.info(f"llm Time to first chunk: {time.perf_counter() - start:.2f}s")
                first = False
            reply += msg
            buf += msg
            # 遇到句號就念；第一句遇到逗號、超過 8 個字就先念，早點開口
            while True:
                m = re.search(r"[。！？!?；;\n]", buf)
                if not m:
                    s = re.search(r"[，,、：:]", buf)
                    if s and s.end() >= (40 if spoken else 8):
                        m = s
                    else:
                        break
                speak(buf[:m.end()])
                spoken = True
                buf = buf[m.end():]
        if buf.strip():
            speak(buf)
        logger.info(f"llm done in {time.perf_counter() - start:.2f}s: {_clean(reply)[:60]}")

        history += [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}]
        del history[:-HISTORY_TURNS * 2]
    except Exception:
        logger.exception('llm exception:')
        avatar_session.put_msg_txt("抱歉，我現在有點恍神，可以再說一次嗎？", datainfo)
