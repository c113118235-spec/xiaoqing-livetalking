/* 小晴嵌入工具：在你的網頁裡放一個會說話的小晴。
 *
 *   <div id="xq" style="width:360px;height:480px"></div>
 *   <script src="https://<小晴的網址>/xiaoqing-embed.js"></script>
 *   <script>
 *     // token 由「你的後端」用 sk_ 金鑰呼叫 POST /api/v1/tokens 換來（金鑰不能放在網頁裡）
 *     const xq = XiaoqingEmbed.mount("#xq", { server: "https://<小晴的網址>", token });
 *     xq.on("ready", () => xq.say("哈囉，我是小晴！"));
 *     xq.on("speaking", (v) => console.log("在說話嗎", v));
 *     // xq.chat("今天要做什麼？")  讓小晴自己回答；xq.interrupt() 打斷；xq.end() 結束
 *   </script>
 */
(function (global) {
  "use strict";
  function mount(target, opts) {
    const el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) throw new Error("XiaoqingEmbed：找不到放小晴的元素");
    if (!opts || !opts.server || !opts.token) throw new Error("XiaoqingEmbed：需要 server 和 token");
    const server = opts.server.replace(/\/$/, "");
    const serverOrigin = new URL(server).origin;
    const frame = document.createElement("iframe");
    frame.src = server + "/embed#" + new URLSearchParams({ token: opts.token, origin: location.origin });
    frame.allow = "autoplay";
    frame.title = "小晴（AI 角色）";
    frame.style.cssText = "border:0;width:100%;height:100%;display:block;background:#0B0A10";
    el.appendChild(frame);

    const handlers = {};
    const pending = {};
    let seq = 0;
    let ready = false;
    const queue = [];
    function fire(type, v) { (handlers[type] || []).forEach((f) => { try { f(v); } catch (e) { console.error(e); } }); }

    function onMsg(e) {
      if (e.origin !== serverOrigin || e.source !== frame.contentWindow) return;
      const m = e.data || {};
      if (!m.xq) return;
      if (m.xq === "ready") { ready = true; queue.splice(0).forEach((x) => frame.contentWindow.postMessage(x, serverOrigin)); fire("ready", m); }
      else if (m.xq === "speaking") fire("speaking", !!m.value);
      else if (m.xq === "ack" && pending[m.id]) { pending[m.id].res(); delete pending[m.id]; }
      else if (m.xq === "error") {
        if (m.id && pending[m.id]) { pending[m.id].rej(new Error(m.message)); delete pending[m.id]; }
        fire("error", m.message);
      } else fire(m.xq, m);                       // playing、needs-gesture、ended
    }
    window.addEventListener("message", onMsg);

    function send(msg) {
      const id = String(++seq);
      return new Promise((res, rej) => {
        pending[id] = { res, rej };
        const full = { ...msg, id };
        if (ready) frame.contentWindow.postMessage(full, serverOrigin);
        else queue.push(full);
      });
    }

    const api = {
      // say("你好", {emotion: "happy", action: "wave", interrupt: true}) 或 say({segments: [{text, emotion, action}, ...]})
      say: (text, o) => send(typeof text === "object" ? { xq: "say", mode: "echo", ...text }
                                                       : { xq: "say", text, mode: "echo", ...(o || {}) }),
      chat: (text, o) => send({ xq: "say", text, mode: "chat", interrupt: !!(o && o.interrupt) }),
      interrupt: () => send({ xq: "interrupt" }),
      end: () => send({ xq: "end" }).finally(() => { window.removeEventListener("message", onMsg); frame.remove(); }),
      on: (type, f) => { (handlers[type] = handlers[type] || []).push(f); return api; },
      iframe: frame,
    };
    return api;
  }
  global.XiaoqingEmbed = { mount };
})(window);
