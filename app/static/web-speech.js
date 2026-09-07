(() => {
  "use strict";

  // Only Czech recognition. Translation, credentials and paid Google STT stay
  // on the server. This controller never promotes a hypothesis to a final.
  window.createSpeechCapture = ({audioTrack, onTranscript, onFirstStt, onFailure, onStopped}) => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let disposed = false;
    let stopping = false;
    let restartTimer = null;
    let stopTimer = null;
    let firstAudioAt = null;
    let firstResult = false;
    let lastResultAt = performance.now();
    let voicedMs = 0;
    let shortRestarts = 0;
    let draft = "";

    function detach() {
      if (!recognition) return;
      const old = recognition;
      recognition = null;
      old.onresult = old.onerror = old.onend = null;
      try { old.abort(); } catch (_) { /* already ended */ }
    }
    function dispose() {
      disposed = true;
      clearTimeout(restartTimer);
      clearTimeout(stopTimer);
      detach();
    }
    function fail(code) {
      if (disposed) return;
      dispose();
      onFailure(code);
    }
    function finish(confirmed) {
      if (disposed) return;
      dispose();
      onStopped(confirmed);
    }
    function start() {
      if (disposed || stopping) return;
      let current;
      try { current = new Recognition(); }
      catch (_) { fail("unavailable"); return; }
      recognition = current;
      current.lang = "cs-CZ";
      current.continuous = true;
      current.interimResults = true;
      current.maxAlternatives = 1;
      const finalIndices = new Set();
      const startedAt = performance.now();
      draft = "";
      current.onresult = event => {
        if (disposed || recognition !== current) return;
        lastResultAt = performance.now();
        voicedMs = 0;
        shortRestarts = 0;
        if (!firstResult && firstAudioAt !== null) {
          firstResult = true;
          onFirstStt(Math.max(0, Math.round(lastResultAt - firstAudioAt)));
        }
        // Result indices are stable within one recognition run. Repeated
        // words in different finals are valid and must never be deduplicated.
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const result = event.results[i];
          if (result.isFinal && !finalIndices.has(i)) {
            finalIndices.add(i);
            const text = result[0].transcript.trim();
            if (text) onTranscript("final", text);
          }
        }
        // The entire non-final suffix is replaceable, including removals.
        const next = Array.from(event.results)
          .filter(result => !result.isFinal)
          .map(result => result[0].transcript.trim()).join(" ").trim();
        if (next !== draft) {
          draft = next;
          if (!stopping) onTranscript("interim", draft);
        }
      };
      current.onerror = event => {
        if (disposed || recognition !== current) return;
        if (stopping) { finish(false); return; }
        // Silence is not a service outage. Permission denial must never be
        // worked around by opening a different microphone/STT path.
        if (event.error !== "no-speech") fail(event.error);
      };
      current.onend = () => {
        if (disposed || recognition !== current) return;
        recognition = null;
        if (stopping) { finish(!draft); return; }
        if (draft) { fail("unfinished"); return; }
        if (performance.now() - startedAt < 1000) shortRestarts += 1;
        else shortRestarts = 0;
        if (shortRestarts >= 3) { fail("unavailable"); return; }
        restartTimer = setTimeout(start, 250);
      };
      // Reuse the already-authorized microphone instead of asking the browser
      // to open a second capture device. Older implementations ignore the
      // optional argument and use their default microphone.
      try { current.start(audioTrack); }
      catch (error) { fail(error.name === "NotAllowedError" ? "not-allowed" : "unavailable"); }
    }
    return {
      start,
      dispose,
      stop() {
        if (disposed || stopping) return;
        stopping = true;
        clearTimeout(restartTimer);
        if (!recognition) { finish(!draft); return; }
        // stop(), not abort(): the browser may deliver its final in onresult
        // before onend. Bound this wait if the recognition service hangs.
        stopTimer = setTimeout(() => finish(false), 3000);
        try { recognition.stop(); } catch (_) { finish(false); }
      },
      audioFrame(buffer) {
        if (disposed || stopping) return;
        const now = performance.now();
        if (firstAudioAt === null) firstAudioAt = now;
        const samples = new DataView(buffer);
        let energy = 0;
        for (let i = 0; i < samples.byteLength; i += 2) {
          energy += (samples.getInt16(i, true) / 32768) ** 2;
        }
        const rms = Math.sqrt(energy / Math.max(1, samples.byteLength / 2));
        if (rms > 0.012) voicedMs += samples.byteLength / 32;
        // Detect a stalled browser service, not a merely quiet microphone.
        if (voicedMs >= 6000 && now - lastResultAt >= 8000) fail("no-results");
      }
    };
  };
})();
