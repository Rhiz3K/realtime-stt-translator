/* Fit the paired preview, never the user's saved font size or final history. */
(() => {
  "use strict";
  window.createInterimLayout = ({panel, heading, workspace, texts, getFontSize, getFollow, onLayout}) => {
    const MIN_FONT_PX = 14;
    const HISTORY_PEEK_PX = 48;
    let frame = null;

    function fit() {
      frame = null;
      const preferred = getFontSize();
      panel.style.setProperty("--live-font-size", `${preferred}px`);
      const area = workspace.getBoundingClientRect();
      const top = panel.getBoundingClientRect().top;
      const viewport = window.visualViewport;
      const bottom = Math.min(area.bottom, viewport
        ? viewport.offsetTop + viewport.height : window.innerHeight);
      const gap = parseFloat(getComputedStyle(workspace).rowGap) || 0;
      // Keep a small, scrollable strip of history accessible. The rest of the
      // workspace belongs to the preview when its content actually needs it.
      const available = Math.max(0, bottom - top - HISTORY_PEEK_PX - gap);
      panel.style.setProperty("--live-max-height", `${Math.floor(available)}px`);

      if (texts.some(text => text.textContent) && panel.dataset.active !== "false" && available > 0) {
        const fits = size => {
          panel.style.setProperty("--live-font-size", `${size}px`);
          return texts.every(text => text.scrollHeight <= text.clientHeight + 1 &&
            text.scrollWidth <= text.clientWidth + 1);
        };
        // CSS grows the panel naturally up to the available height first.
        // Only overflow at that limit permits reducing the preview font.
        if (!fits(preferred)) {
          let low = MIN_FONT_PX;
          let high = preferred - 1;
          let best = MIN_FONT_PX;
          while (low <= high) {
            const candidate = Math.floor((low + high) / 2);
            if (fits(candidate)) { best = candidate; low = candidate + 1; }
            else high = candidate - 1;
          }
          panel.style.setProperty("--live-font-size", `${best}px`);
        }
        // At the readable floor, exceptionally long text stays available via
        // scrolling. Never truncate it or force scrolling when follow is off.
        if (getFollow()) for (const text of texts) text.scrollTop = text.scrollHeight;
      }
      onLayout();
    }
    function schedule() {
      if (frame === null) frame = window.requestAnimationFrame(fit);
    }
    function reset() {
      panel.style.setProperty("--live-font-size", `${getFontSize()}px`);
      for (const text of texts) text.scrollTop = 0;
      schedule();
    }

    // Observe only geometry independent of the fitted text. Observing the
    // panel itself would create resize/fit feedback loops as its font changes.
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(schedule);
      observer.observe(workspace);
      observer.observe(heading);
    }
    window.addEventListener("resize", schedule);
    window.visualViewport?.addEventListener("resize", schedule);
    window.visualViewport?.addEventListener("scroll", schedule);
    document.addEventListener("fullscreenchange", schedule);
    document.fonts?.ready.then(schedule);
    return {schedule, reset};
  };
})();
