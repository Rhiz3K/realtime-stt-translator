/* Presentation only. This layer cannot choose models or send audio/config. */
(() => {
  "use strict";
  window.createTranslatorUI = () => {
    const el = (id) => document.getElementById(id);
    const root = document.documentElement;
    const history = el("history");
    const historyScroll = el("history-scroll");
    const livePanel = el("live-panel");
    const liveEn = el("live-en");
    const liveRu = el("live-ru");
    const liveNote = el("live-note");
    const template = el("result-template");
    const fontSize = el("font-size");
    const follow = el("follow-latest");
    const full = el("fullscreen-toggle");
    const compact = el("compact-toggle");
    const theme = el("theme-toggle");
    const storageKey = "translator.presentation.v1";
    const prefs = {font: 28, compact: false, follow: true, theme: "system"};
    let totalFinals = 0;
    let interimCount = 0;
    let finalCount = 0;
    let audioBytes = 0;
    let connectingAt = null;
    let active = false;
    let lastPreviewAt = null;
    let previewWarning = false;
    let lastHistoryScrollTop = 0;

    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (saved && typeof saved === "object") {
        if (Number.isFinite(saved.font)) prefs.font = Math.max(18, Math.min(64, Math.round(saved.font / 2) * 2));
        if (typeof saved.compact === "boolean") prefs.compact = saved.compact;
        if (typeof saved.follow === "boolean") prefs.follow = saved.follow;
        if (["system", "dark", "light"].includes(saved.theme)) prefs.theme = saved.theme;
      }
    } catch (_) { /* Storage may be blocked, full, or contain invalid JSON. */ }

    function save() {
      try { localStorage.setItem(storageKey, JSON.stringify(prefs)); } catch (_) { /* Optional preference only. */ }
    }
    function notice(message) {
      el("presentation-notice").textContent = message;
      el("presentation-notice").hidden = !message;
    }
    function applyFont(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      prefs.font = Math.max(18, Math.min(64, Math.round(number / 2) * 2));
      root.style.setProperty("--translation-size", `${prefs.font}px`);
      fontSize.value = String(prefs.font);
      el("font-value").textContent = `${prefs.font} px`;
      el("font-down").disabled = prefs.font === 18;
      el("font-up").disabled = prefs.font === 64;
      followLatest();
    }
    function applyCompact() {
      el("toolbar").dataset.compact = String(prefs.compact);
      compact.setAttribute("aria-pressed", String(prefs.compact));
      compact.setAttribute("aria-expanded", String(!prefs.compact));
      compact.setAttribute("aria-controls", "toolbar-details");
      const label = prefs.compact ? "Rozbalit horní lištu" : "Sbalit horní lištu";
      compact.setAttribute("aria-label", label);
      compact.title = label;
      followLatest();
    }
    function darkTheme() {
      return prefs.theme === "dark" || (prefs.theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    }
    function applyTheme() {
      if (prefs.theme === "system") delete root.dataset.theme;
      else root.dataset.theme = prefs.theme;
      const dark = darkTheme();
      theme.setAttribute("aria-pressed", String(dark));
      theme.setAttribute("aria-label", dark ? "Přepnout na světlý motiv" : "Přepnout na tmavý motiv");
      theme.title = dark ? "Světlý motiv" : "Tmavý motiv";
    }
    function syncFullscreen() {
      const enabled = !!document.fullscreenElement;
      full.setAttribute("aria-pressed", String(enabled));
      full.setAttribute("aria-label", enabled ? "Ukončit celou obrazovku" : "Celá obrazovka");
      full.title = enabled ? "Ukončit celou obrazovku (Esc)" : "Celá obrazovka";
    }
    function followLatest() {
      if (!prefs.follow) return;
      historyScroll.scrollTop = 0;
      lastHistoryScrollTop = historyScroll.scrollTop;
    }
    function prependHistory(fragment) {
      const previousHeight = historyScroll.scrollHeight;
      const previousTop = historyScroll.scrollTop;
      const wasEmpty = history.children.length === 0;
      history.prepend(fragment);
      el("history-empty").hidden = true;
      // CSS scroll anchoring is disabled here: preserve the reader's place
      // exactly once when new rows are inserted above older text.
      historyScroll.scrollTop = prefs.follow || wasEmpty
        ? 0
        : previousTop + Math.max(0, historyScroll.scrollHeight - previousHeight);
      lastHistoryScrollTop = historyScroll.scrollTop;
    }
    function phase(text, state = "live") {
      liveNote.textContent = text;
      livePanel.dataset.state = state;
    }
    function clearInterim() {
      liveEn.textContent = "";
      liveRu.textContent = "";
      lastPreviewAt = null;
      previewWarning = false;
      phase(active ? "Čekám na další řeč…" : "Připraveno na českou řeč", active ? "live" : "idle");
    }
    function renderInterim(en, ru) {
      // Both writes happen in one synchronous task, before the next paint.
      liveEn.textContent = en;
      liveRu.textContent = ru;
      lastPreviewAt = performance.now();
      previewWarning = false;
      phase("Živý náhled · text se může ještě změnit");
      el("stat-interim").textContent = String(++interimCount);
      if (prefs.follow) {
        liveEn.scrollTop = liveEn.scrollHeight;
        liveRu.scrollTop = liveRu.scrollHeight;
      }
      // Preview growth changes the history viewport height. Follow after
      // layout, so the newest committed paragraph is still fully reachable.
      followLatest();
    }
    function renderFinal(en, ru) {
      const fragment = template.content.cloneNode(true);
      fragment.querySelector(".result-en").textContent = en;
      fragment.querySelector(".result-ru").textContent = ru;
      totalFinals += 1;
      el("history-count").textContent = String(totalFinals);
      el("stat-final").textContent = String(++finalCount);
      clearInterim();
      prependHistory(fragment);
    }
    function showError(code, message) {
      if (code === "interim_translation_failed") {
        previewWarning = true;
        phase("Interim se nepodařil · čekám na další náhled nebo final", "error");
        return; // Keep last preview and microphone; this is not a missing final.
      }
      if (["translation_failed", "transcription_incomplete", "audio_backpressure"].includes(code)) {
        const gap = document.createElement("article");
        gap.className = "result result-error";
        gap.dataset.code = code;
        gap.setAttribute("role", "status");
        gap.textContent = message;
        clearInterim();
        prependHistory(gap);
      }
      phase(message, "error");
    }
    function ms(id, value) {
      if (Number.isFinite(value) && value >= 0) el(id).textContent = `${Math.round(value)} ms`;
    }
    function telemetry(message) {
      ms("stat-stt", message.first_stt_ms);
      ms("stat-translation", message.translation_ms);
      ms("stat-queue", message.queue_ms);
      if (Number.isInteger(message.hypotheses) && Number.isInteger(message.finals_received)) {
        el("stat-hypotheses").textContent = `${message.hypotheses} / ${message.finals_received}`;
      }
      if (message.stage === "translating") {
        if (message.kind === "final") phase("Dokončuji finální překlad…", "busy");
        else if (!previewWarning) {
          const age = lastPreviewAt === null ? "" : ` · náhled před ${Math.floor((performance.now() - lastPreviewAt) / 1000)} s`;
          phase(`Překládám průběžnou řeč…${age}`, "busy");
        }
      }
    }
    function beginSession() {
      active = true;
      interimCount = 0; finalCount = 0; audioBytes = 0;
      for (const id of ["stat-interim", "stat-final"]) el(id).textContent = "0";
      for (const id of ["stat-stt", "stat-translation", "stat-queue", "stat-connect"]) el(id).textContent = "—";
      el("stat-audio").textContent = "00:00";
      el("stat-hypotheses").textContent = "0 / 0";
      el("audio-level").value = 0;
      clearInterim();
    }
    function endSession() {
      active = false;
      el("audio-level").value = 0;
      clearInterim();
    }
    function audioFrame(buffer) {
      audioBytes += buffer.byteLength;
      const seconds = Math.floor(audioBytes / 32000);
      el("stat-audio").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
      // Measure the exact captured PCM, without a second audio graph or stream.
      const pcm = new DataView(buffer);
      let energy = 0;
      for (let index = 0; index < buffer.byteLength; index += 2) energy += (pcm.getInt16(index, true) / 32768) ** 2;
      const rms = Math.sqrt(energy / Math.max(1, buffer.byteLength / 2));
      el("audio-level").value = Math.max(0, Math.min(100, (20 * Math.log10(Math.max(rms, 0.00001)) + 60) / 60 * 100));
    }

    fontSize.addEventListener("input", () => { applyFont(fontSize.value); save(); });
    el("font-down").addEventListener("click", () => { applyFont(prefs.font - 2); save(); });
    el("font-up").addEventListener("click", () => { applyFont(prefs.font + 2); save(); });
    follow.addEventListener("change", () => { prefs.follow = follow.checked; save(); followLatest(); });
    historyScroll.addEventListener("scroll", () => {
      if (prefs.follow && historyScroll.scrollTop > lastHistoryScrollTop && historyScroll.scrollTop > 80) {
        prefs.follow = false; follow.checked = false; save();
      }
      lastHistoryScrollTop = historyScroll.scrollTop;
    }, {passive: true});
    compact.addEventListener("click", () => { prefs.compact = !prefs.compact; applyCompact(); save(); });
    theme.addEventListener("click", () => { prefs.theme = darkTheme() ? "light" : "dark"; applyTheme(); save(); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
    full.addEventListener("click", async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await root.requestFullscreen();
        notice("");
      } catch (_) { notice("Prohlížeč nepovolil celou obrazovku. Zkuste klávesu F11."); }
      syncFullscreen();
    });
    document.addEventListener("fullscreenchange", syncFullscreen);
    if (!root.requestFullscreen || document.fullscreenEnabled === false) {
      full.disabled = true;
      full.title = "Celá obrazovka není v tomto prohlížeči dostupná";
    }
    applyFont(prefs.font); applyCompact(); applyTheme(); syncFullscreen();
    follow.checked = prefs.follow;

    return {
      clearInterim, renderInterim, renderFinal, showError, telemetry, beginSession, endSession, audioFrame,
      connecting() { connectingAt = performance.now(); },
      connected() { if (connectingAt !== null) ms("stat-connect", performance.now() - connectingAt); phase("Poslouchám · čekám na první průběžný překlad…"); }
    };
  };
})();
