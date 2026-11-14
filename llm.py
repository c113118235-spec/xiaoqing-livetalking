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
    f"從現在開始你扮演「{NAME}」，是使用者的虛擬女朋友，一律用台灣繁體中文和使用者對話，絕對不要用簡體字。"
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
ACT_MAP = {"站起來": "stand", "站起来": "stand", "坐下": "sit", "揮手": "wave", "挥手": "wave",
           "比讚": "thumbs_up", "比赞": "thumbs_up", "拍手": "clap", "鼓掌": "clap"}
ACT_TAG = re.compile(r"[【\[]\s*(" + "|".join(ACT_MAP) + r")\s*[】\]]")
ACT_NAME = {"stand": "站起來", "sit": "坐下", "wave": "揮手", "thumbs_up": "比讚", "clap": "拍手"}
# 使用者這樣說時，不等 LLM，直接做動作（LLM 不一定會記得加標籤）
ACT_KEYWORDS = {
    "wave": re.compile(r"掰掰|拜拜|再見|晚安|去睡|先走了|下次見|揮手|揮揮手|打個招呼|嗨|哈囉|bye|hello|\bhi\b", re.I),
    "stand": re.compile(r"站起來|起來一下"),
    "sit": re.compile(r"坐下|坐回去"),
    # 對方分享好消息、完成事情時比讚
    "thumbs_up": re.compile(r"比讚|按讚|考過|考好|及格|錄取|上榜|得獎|做完了|完成了|搞定|成功了|升職|加薪|我做到了|好消息"),
    "clap": re.compile(r"拍手|鼓掌|拍拍手|恭喜|生日快樂|太棒了|慶祝|新年快樂|聖誕快樂"),
}
WEEKDAYS = "一二三四五六日"


def _available_actions():
    try:
        from server import gf_pose
        return list(gf_pose.actions().get("actions", {}))
    except Exception:
        return []


def _action_prompt(acts):
    if not acts:
        return ""
    tags = "".join(f"【{ACT_NAME[a]}】" for a in acts if a in ACT_NAME)
    return ("你可以做動作：打招呼、道別，或對方請你做動作時，在那句話的情緒標籤後面加一個動作標籤，"
            f"只能從{tags}裡選，一次回答最多一個，不要每次都用。")


PERSONA_FILE = "data/personas.json"


def _personas() -> dict:
    """讀 data/personas.json（每次都重讀，改檔案不用重開）"""
    import json
    try:
        with open(PERSONA_FILE, encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d.get("personas"), dict) and d["personas"]:
                return d
    except (OSError, ValueError):
        pass
    return {"default": "girlfriend", "personas": {"girlfriend": {"label": "女友"}}}


def default_persona_id() -> str:
    d = _personas()
    pid = d.get("default", "girlfriend")
    return pid if pid in d["personas"] else next(iter(d["personas"]))


def persona_list() -> dict:
    d = _personas()
    return {"default": default_persona_id(),
            "personas": [{"id": k, "label": v.get("label", k), "board": bool(v.get("board"))}
                         for k, v in d["personas"].items()]}


def _persona_cfg(pid) -> dict:
    return _personas()["personas"].get(pid or "", {}) or {}


def _persona(pid=None, name=None) -> str:
    p = _persona_cfg(pid).get("prompt")
    if p:
        return p.replace("{name}", name or NAME)
    try:
        with open("persona.txt", encoding="utf-8") as f:
            p = f.read().strip()
            if p:
                return p
    except OSError:
        pass
    return _DEFAULT_PERSONA


def _examples(pid=None) -> list:
    ex = _persona_cfg(pid).get("examples")
    if not ex:
        return list(EXAMPLES)
    return [{"role": e["role"], "content": e["content"].replace("{name}", NAME)} for e in ex]


def set_persona(avatar_session, pid):
    """切換這個連線的人設：清空對話紀錄，回傳 (ok, label, 打招呼的話)"""
    d = _personas()
    if pid not in d["personas"]:
        return False, "", ""
    cfg = d["personas"][pid]
    greeting = (cfg.get("greeting") or "").replace("{name}", getattr(avatar_session, "_gf_name", None) or NAME)
    avatar_session._gf_persona = pid
    avatar_session._gf_history = [{"role": "assistant", "content": greeting}] if greeting else []
    return True, cfg.get("label", pid), greeting


def _board_prompt(avatar_session, pid) -> str:
    """學伴黑板的內容（personas.json 裡 "board": true 的人設才會加）"""
    if not _persona_cfg(pid).get("board"):
        return ""
    try:
        from server import gf_board
        return gf_board.board_prompt(avatar_session)
    except Exception:
        logger.exception("board prompt 失敗")
        return ""


def strip_tags(text: str) -> str:
    return ACT_TAG.sub("", EMO_TAG.sub("", text)).strip()


def _now_text() -> str:
    t = datetime.now(ZoneInfo("Asia/Taipei"))
    part = ("凌晨" if t.hour < 5 else "早上" if t.hour < 11 else "中午" if t.hour < 13
            else "下午" if t.hour < 18 else "晚上" if t.hour < 23 else "深夜")
    return f"現在是台灣時間星期{WEEKDAYS[t.weekday()]}{part} {t:%H:%M}。"


# LLM 偶爾會冒出簡體字：念出來、字幕都先轉成台灣繁體
# 優先用 OpenCC（s2twp：連用語都轉成台灣說法），沒裝就用 zhconv，再沒有就用下面的常用字對照表
_S2T_PAIRS = ("没沒 时時 饭飯 吗嗎 这這 说說 么麼 个個 们們 对對 会會 来來 还還 过過 样樣 为為 后後 让讓 问問 题題 学學 习習 点點 "
              "儿兒 里裡 发發 开開 关關 从從 见見 觉覺 头頭 话話 谁誰 难難 爱愛 欢歡 长長 应應 该該 实實 现現 经經 东東 车車 "
              "边邊 门門 间間 听聽 声聲 认認 识識 记記 给給 请請 谢謝 写寫 读讀 书書 课課 试試 题題 讲講 语語 词詞 网網 "
              "电電 脑腦 机機 视視 乐樂 动動 运運 场場 处處 气氣 热熱 冷冷 饿餓 饱飽 药藥 医醫 钱錢 买買 卖賣 贵貴 "
              "岁歲 年年 节節 礼禮 颜顏 体體 脸臉 双雙 简簡 单單 复複 杂雜 业業 务務 办辦 准準 备備 计計 划劃 华華 "
              "国國 际際 区區 级級 帮幫 转轉 换換 别別 总總 还還 尽盡 虽雖 然然 却卻 带帶 带帶 张張 万萬 两兩 几幾 "
              "么麼 够夠 累累 啰囉 吧吧 呢呢 啊啊 嘛嘛 呀呀 哦哦 么麼 什什 怎怎 为為 们們 让讓 晚晚 睡睡 觉覺 梦夢")
_S2T = {p[0]: p[1] for p in _S2T_PAIRS.split() if len(p) == 2}
_to_tw_impl = None


def _to_tw(t: str) -> str:
    global _to_tw_impl
    if _to_tw_impl is None:
        try:
            import opencc
            cc = opencc.OpenCC("s2twp")
            _to_tw_impl = cc.convert
        except Exception:
            try:
                import zhconv
                _to_tw_impl = lambda x: zhconv.convert(x, "zh-tw")
            except Exception:
                logger.info("llm 簡轉繁：沒有 opencc／zhconv，用內建常用字對照表")
                _to_tw_impl = lambda x: "".join(_S2T.get(ch, ch) for ch in x)
    try:
        return _to_tw_impl(t)
    except Exception:
        return t


def _clean(t: str) -> str:
    t = EMO_TAG.sub("", t)
    t = ACT_TAG.sub("", t)
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
        acts = _available_actions() if getattr(avatar_session, "_gf_pose", None) is not None else []   # 自訂學伴沒有動作
        pid = getattr(avatar_session, "_gf_persona", None) or default_persona_id()
        examples = _examples(pid)
        if "wave" in acts:
            examples += [{"role": "user", "content": "我要去睡了，掰掰"},
                         {"role": "assistant", "content": "【溫柔】【揮手】晚安，祝你做個好夢！【開心】明天見喔。"}]
        if "thumbs_up" in acts:
            examples += [{"role": "user", "content": "我期中考考得不錯耶"},
                         {"role": "assistant", "content": "【開心】【比讚】太厲害了吧！【溫柔】你最近那麼努力，真的值得。"}]
        messages = ([{'role': 'system', 'content': _persona(pid, getattr(avatar_session, "_gf_name", None)) + _action_prompt(acts) + _board_prompt(avatar_session, pid) + _now_text()}] + examples + history +
                    [{'role': 'user', 'content': message}])

        # 關鍵字直接觸發動作（例如說「掰掰」就揮手），不用等 LLM
        ctl = getattr(avatar_session, "_gf_pose", None)
        if ctl is not None:
            for act in sorted(acts, key=lambda a: a != "thumbs_up"):   # 好消息優先比讚
                kw = ACT_KEYWORDS.get(act)
                if kw and kw.search(message):
                    ok, why = ctl.request(act)
                    logger.info(f"關鍵字動作 {act}：{why}")
                    break
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
            # 動作標籤：請姿勢控制做動作，等動作做完再說這句（最多等 8 秒）
            ctl = getattr(avatar_session, "_gf_pose", None)
            for act in ACT_TAG.findall(text):
                if ctl is not None:
                    ok, why = ctl.request(ACT_MAP[act])
                    logger.info(f"llm 動作 {act}：{why}")
            if ctl is not None:
                t0 = time.time()
                while ctl.busy() and time.time() - t0 < 8:
                    time.sleep(0.05)
            t = _to_tw(_clean(text))
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
