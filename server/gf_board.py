"""學伴黑板：每個連線各自的讀書進度（今日目標、作業、截止日）。

- 黑板內容存在使用者自己的瀏覽器，每次變動就 POST /board 同步到這個連線（只放在記憶體）
- 學伴模式的 LLM 會看到黑板（board_prompt），被問到進度時照黑板回答、偶爾提醒快到期的作業
- POST /board/say 讓學伴主動開口：remind = 提醒最急的作業，done = 稱讚完成了一項
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from aiohttp import web

logger = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Taipei")
MAX_TASKS = 30
MAX_TEXT = 60
WEEKDAYS = "一二三四五六日"


# ───────────── 黑板資料 ─────────────

def _today() -> date:
    return datetime.now(TZ).date()


def _clean_text(s, n=MAX_TEXT) -> str:
    s = re.sub(r"[\x00-\x1f【】<>]", "", str(s or "")).strip()   # 拿掉控制字元和會干擾情緒標籤的括號
    return s[:n]


def _parse_due(s):
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def _normalize(data) -> dict:
    goal = _clean_text((data or {}).get("goal", ""), 80)
    tasks = []
    for t in ((data or {}).get("tasks") or [])[:MAX_TASKS]:
        text = _clean_text(t.get("text"))
        if not text:
            continue
        due = _parse_due(t.get("due"))
        tasks.append({"text": text, "due": due.isoformat() if due else "", "done": bool(t.get("done"))})
    return {"goal": goal, "tasks": tasks}


def _days_left(t) -> int | None:
    d = _parse_due(t.get("due"))
    return None if d is None else (d - _today()).days


def _when(days: int | None, due: str) -> str:
    if days is None:
        return "沒有截止日"
    d = _parse_due(due)
    md = f"{d.month}/{d.day}（星期{WEEKDAYS[d.weekday()]}）"
    if days < 0:
        return f"{md}，已經逾期 {-days} 天"
    if days == 0:
        return f"{md}，就是今天"
    if days == 1:
        return f"{md}，明天"
    if days == 2:
        return f"{md}，後天"
    return f"{md}，還有 {days} 天"


def _urgent(board) -> dict | None:
    """最急的未完成作業：逾期 > 今天 > 明天 > 三天內"""
    best, best_days = None, None
    for t in board.get("tasks", []):
        if t["done"]:
            continue
        days = _days_left(t)
        if days is None or days > 3:
            continue
        if best is None or days < best_days:
            best, best_days = t, days
    return best


def board_prompt(avatar_session) -> str:
    """給 LLM 的黑板內容（學伴模式才會用到）"""
    board = getattr(avatar_session, "_gf_board", None)
    if not board or (not board["goal"] and not board["tasks"]):
        return ("\n使用者的讀書黑板目前是空的。如果對方提到作業或考試，可以建議他寫到右邊的黑板上，"
                "你會幫他記著、提醒他。")
    t = _today()
    lines = [f"\n以下是使用者寫在讀書黑板上的進度（今天是 {t.month}/{t.day} 星期{WEEKDAYS[t.weekday()]}）："]
    if board["goal"]:
        lines.append(f"今日目標：{board['goal']}")
    todo = [x for x in board["tasks"] if not x["done"]]
    done = [x for x in board["tasks"] if x["done"]]
    todo.sort(key=lambda x: (_days_left(x) is None, _days_left(x) or 0))
    if todo:
        lines.append("還沒完成：" + "；".join(f"{x['text']}（{_when(_days_left(x), x['due'])}）" for x in todo))
    if done:
        lines.append("已經完成：" + "、".join(x["text"] for x in done))
    lines.append(f"進度：完成 {len(done)} / {len(board['tasks'])} 項。")
    lines.append("對方問起進度、作業或今天要做什麼時，照黑板回答，從最急的開始講。"
                 "平常聊天時，如果有逾期或快到期、還沒完成的項目，可以偶爾自然地提醒一句，不要每次都提。"
                 "黑板上沒有的作業不要自己編。")
    return "\n".join(lines)


# ───────────── 學伴主動開口 ─────────────

def _remind_line(t) -> str:
    """提醒的句子：不管黑板上寫的是作業、考試還是開會，都要講得通"""
    days = _days_left(t)
    name = f"「{t['text']}」"
    if days < 0:
        return f"【溫柔】{name}的日期已經過了 {-days} 天喔。【平常】還沒完成的話，我們現在先處理好不好？"
    if days == 0:
        return f"【平常】提醒你一下，{name}就是今天喔。【溫柔】準備好了嗎？還沒的話我陪你。"
    if days == 1:
        return f"【平常】{name}是明天喔。【溫柔】今天先準備一點，明天會輕鬆很多。"
    return f"【平常】再 {days} 天就是{name}了。【開心】現在開始準備剛剛好，不會趕。"


def _summary_line(board) -> str:
    todo = [x for x in board.get("tasks", []) if not x["done"]]
    if not board.get("tasks"):
        return "【平常】黑板上還沒有作業喔。【開心】把要做的事寫上去，我幫你記著。"
    if not todo:
        return "【開心】黑板上的事全部做完了，好厲害！【溫柔】今天可以好好休息一下。"
    first = todo[0]["text"]
    return f"【平常】最近幾天沒有快到期的項目，還剩 {len(todo)} 項沒做完。【開心】要不要先從「{first}」開始？"


def _session(sid):
    from server.session_manager import session_manager
    s = session_manager.get_session(str(sid or ""))
    return s if s is not None else session_manager.get_session(sid)


def _is_study(s) -> bool:
    import llm
    pid = getattr(s, "_gf_persona", None) or llm.default_persona_id()
    return bool(llm._persona_cfg(pid).get("board"))


def _speak(s, line: str) -> None:
    import llm
    text = llm.strip_tags(line)
    if text and hasattr(s, "put_msg_txt"):
        s.put_msg_txt(text)
    hist = getattr(s, "_gf_history", None)
    if hist is not None:
        hist.append({"role": "assistant", "content": line})


async def board(request):
    """POST /board {sessionid, goal, tasks:[{text, due, done}]}：同步黑板"""
    try:
        params = await request.json()
        s = _session(params.get("sessionid"))
        if s is None:
            return web.json_response({"ok": False, "msg": "找不到這個連線"}, status=400)
        s._gf_board = _normalize(params)
        return web.json_response({"ok": True, "count": len(s._gf_board["tasks"])})
    except Exception as e:
        logger.exception("[gf-board] /board exception:")
        return web.json_response({"ok": False, "msg": str(e)}, status=500)


async def board_say(request):
    """POST /board/say {sessionid, event: remind|done, manual, text}：讓學伴開口"""
    try:
        params = await request.json()
        s = _session(params.get("sessionid"))
        if s is None:
            return web.json_response({"ok": False, "msg": "找不到這個連線"}, status=400)
        if not _is_study(s):
            return web.json_response({"ok": False, "msg": "不是學伴模式"})
        b = getattr(s, "_gf_board", None) or {"goal": "", "tasks": []}
        event = params.get("event")
        if event == "done":
            name = _clean_text(params.get("text"))
            left = sum(1 for x in b["tasks"] if not x["done"])
            line = (f"【開心】「{name}」完成了，太棒了！" +
                    ("【溫柔】黑板上的事全部做完了，辛苦了。" if left == 0 else f"【平常】還剩 {left} 項，繼續加油。"))
        elif event == "remind":
            t = _urgent(b)
            if t is None and not params.get("manual"):
                return web.json_response({"ok": False, "msg": "沒有快到期的作業"})
            line = _remind_line(t) if t else _summary_line(b)
        else:
            return web.json_response({"ok": False, "msg": "不支援的事件"}, status=400)
        _speak(s, line)
        logger.info("[gf-board] %s %s", event, llm_safe(line))
        return web.json_response({"ok": True})
    except Exception as e:
        logger.exception("[gf-board] /board/say exception:")
        return web.json_response({"ok": False, "msg": str(e)}, status=500)


def llm_safe(s: str) -> str:
    return s[:60]


def setup(app):
    app.router.add_post("/board", board)
    app.router.add_post("/board/say", board_say)
