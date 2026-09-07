(() => {
  "use strict";

  const toggle = document.getElementById("toggle");
  const status = document.getElementById("status");
  const ui = window.createTranslatorUI();
  const {clearInterim, renderInterim, renderFinal} = ui;
  // Half a second of PCM. Abort explicitly before latency can grow into
  // seconds or a recovered connection can exceed the server's burst limit.
  const MAX_BUFFERED_AUDIO_BYTES = 16000;

  let socket = null;
  let mediaStream = null;
  let audioContext = null;
  let sourceNode = null;
  let captureNode = null;
  let silentGain = null;
  let running = false;
  let starting = false;
  let stopping = false;
  let stopSent = false;
  let generation = 0;
  let flushTimer = null;
  let errorShown = false;

  function setStatus(text, state = "idle") {
    status.textContent = text;
    status.dataset.state = state;
  }

  function setButton(active, disabled = false) {
    toggle.dataset.active = active ? "true" : "false";
    toggle.textContent = active ? "Zastavit" : "Spustit mikrofon";
    toggle.disabled = disabled;
  }

  function showError(code) {
    const messages = {
      audio_too_large: "Zvukový rámec je příliš velký.",
      audio_rate_exceeded: "Zvuk přicházel rychleji než v reálném čase.",
      invalid_audio: "Prohlížeč odeslal neplatný zvuk.",
      server_busy: "Server právě obsluhuje maximum živých relací.",
      google_rate_limited: "Google je právě vytížený. Zkuste to za chvíli.",
      google_timeout: "Google neodpověděl včas.",
      google_unavailable: "Google překlad je dočasně nedostupný.",
      interim_translation_failed: "Průběžný překlad je dočasně nedostupný. Finální překlad pokračuje.",
      translation_failed: "Jeden konečný překlad se nepodařilo vytvořit.",
      transcription_incomplete: "Úplnost závěru řeči se nepodařilo potvrdit. Překlad může být neúplný.",
      audio_backpressure: "Síť nestíhá přenášet zvuk. Relace byla zastavena; závěr může chybět.",
      translation_backlog_full: "Překlad nestíhá přijímanou řeč. Spusťte novou relaci.",
      session_limit: "Relace dosáhla časového limitu. Po dokončení překladů spusťte novou.",
      audio_unavailable: "Tento prohlížeč neumí bezpečně zpracovat mikrofon.",
      session_failed: "Relace byla přerušena. Zkuste ji spustit znovu."
    };
    errorShown = true;
    const message = messages[code] || "Překlad se nepodařilo dokončit.";
    setStatus(message, "error");
    ui.showError(code, message);
  }

  async function releaseAudio() {
    if (sourceNode) sourceNode.disconnect();
    if (captureNode) {
      captureNode.port.onmessage = null;
      captureNode.port.close();
      captureNode.disconnect();
    }
    if (silentGain) silentGain.disconnect();
    sourceNode = null;
    captureNode = null;
    silentGain = null;
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    const closingContext = audioContext;
    audioContext = null;
    if (closingContext && closingContext.state !== "closed") {
      try { await closingContext.close(); } catch (_) { /* already closing */ }
    }
  }

  async function reset(closeSocket = true) {
    const resetGeneration = ++generation;
    clearTimeout(flushTimer);
    flushTimer = null;
    const closingSocket = socket;
    socket = null;
    stopping = true;
    setButton(false, true);
    clearInterim();
    ui.endSession();
    const released = releaseAudio();
    if (closeSocket && closingSocket && closingSocket.readyState < WebSocket.CLOSING) {
      closingSocket.close(1000, "Client reset");
    }
    await released;
    if (resetGeneration !== generation) return;
    running = false;
    starting = false;
    stopping = false;
    stopSent = false;
    setButton(false, false);
    if (!errorShown) setStatus("Připraveno");
  }

  function abortCongestedSession(run, activeSocket) {
    if (run !== generation || socket !== activeSocket || stopSent) return;
    showError("audio_backpressure");
    stopSent = true;
    const released = reset(false);
    activeSocket.close(4000, "Audio network backlog");
    void released;
  }

  function sendStop(run, activeSocket) {
    if (run !== generation || socket !== activeSocket || stopSent) return;
    if (activeSocket.bufferedAmount > MAX_BUFFERED_AUDIO_BYTES) {
      abortCongestedSession(run, activeSocket);
      return;
    }
    stopSent = true;
    clearTimeout(flushTimer);
    flushTimer = null;
    if (activeSocket.readyState === WebSocket.OPEN) {
      activeSocket.send(JSON.stringify({type: "stop"}));
    } else {
      void reset();
    }
  }

  async function beginCapture(run) {
    if (run !== generation || stopping || captureNode) return;
    if (!audioContext || !mediaStream) throw new Error("Audio capture is not ready.");
    const activeSocket = socket;
    const activeContext = audioContext;

    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(audioContext, "pcm16-capture");
    const activeCapture = captureNode;
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    captureNode.port.onmessage = (event) => {
      if (run !== generation || socket !== activeSocket || captureNode !== activeCapture) return;
      if (event.data instanceof ArrayBuffer) {
        if (activeSocket.readyState === WebSocket.OPEN && !stopSent) {
          if (activeSocket.bufferedAmount + event.data.byteLength > MAX_BUFFERED_AUDIO_BYTES) {
            abortCongestedSession(run, activeSocket);
            return;
          }
          activeSocket.send(event.data);
          ui.audioFrame(event.data);
        }
      } else if (event.data && event.data.type === "flushed" && stopping) {
        sendStop(run, activeSocket);
      }
    };
    sourceNode.connect(captureNode);
    captureNode.connect(silentGain);
    silentGain.connect(audioContext.destination);
    if (activeContext.state !== "running") await activeContext.resume();
    if (run !== generation || socket !== activeSocket || stopping) return;
    ui.connected();
    running = true;
    starting = false;
    setButton(true, false);
    setStatus("Poslouchám…", "live");
  }

  async function start() {
    if (starting || running || stopping) return;
    const run = ++generation;
    starting = true;
    errorShown = false;
    ui.beginSession();
    clearInterim();
    setButton(false, true);
    setStatus("Čekám na mikrofon…");
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass || !window.AudioWorkletNode) {
        throw new Error("AudioWorklet není v tomto prohlížeči dostupný.");
      }
      try {
        audioContext = new AudioContextClass({sampleRate: 16000, latencyHint: "interactive"});
      } catch (_) {
        audioContext = new AudioContextClass({latencyHint: "interactive"});
      }

      // Start/resume while the click still carries user activation. Safari may
      // reject a first resume attempted later from a WebSocket callback.
      const audioReady = Promise.all([
        audioContext.resume(),
        audioContext.audioWorklet.addModule("/static/pcm-worklet.js")
      ]);
      const microphoneReady = navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      }).then((stream) => {
        if (run !== generation || !starting) {
          stream.getTracks().forEach((track) => track.stop());
          return null;
        }
        mediaStream = stream;
        return stream;
      });
      const [, acquiredStream] = await Promise.all([audioReady, microphoneReady]);
      if (run !== generation || !acquiredStream) return;
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      ui.connecting();
      const activeSocket = new WebSocket(`${protocol}//${location.host}/ws/audio`);
      socket = activeSocket;
      activeSocket.binaryType = "arraybuffer";
      activeSocket.addEventListener("open", () => {
        if (run === generation && socket === activeSocket) {
          setStatus("Připojuji Google…");
        }
      });
      activeSocket.addEventListener("message", async (event) => {
        if (run !== generation || socket !== activeSocket) return;
        let message;
        try { message = JSON.parse(event.data); } catch (_) { return; }
        if (!message || typeof message.type !== "string") return;
        if (message.type === "ready") {
          try { await beginCapture(run); }
          catch (error) {
            if (run !== generation || socket !== activeSocket) return;
            showError("audio_unavailable");
            await reset();
          }
        } else if (
          (message.type === "interim" || message.type === "final") &&
          typeof message.en === "string" && typeof message.ru === "string"
        ) {
          if (message.type === "interim") renderInterim(message.en, message.ru);
          else renderFinal(message.en, message.ru);
          if (running && !stopping) {
            errorShown = false;
            setStatus("Poslouchám…", "live");
          }
        } else if (message.type === "telemetry") {
          ui.telemetry(message);
        } else if (message.type === "error") {
          showError(message.code);
          if (!["translation_failed", "interim_translation_failed"].includes(message.code)) {
            stopping = true;
            stopSent = true;
            clearTimeout(flushTimer);
            flushTimer = null;
            setButton(running, true);
            await releaseAudio();
          }
        } else if (message.type === "ended") {
          if (!errorShown) setStatus("Dokončeno");
        }
      });
      activeSocket.addEventListener("error", () => {
        if (run === generation && socket === activeSocket) {
          showError("session_failed");
        }
      });
      activeSocket.addEventListener("close", () => {
        if (socket === activeSocket) void reset(false);
      });
    } catch (error) {
      if (run !== generation) return;
      generation += 1;
      errorShown = true;
      setStatus("Mikrofon není dostupný nebo nebyl povolen.", "error");
      await reset();
    }
  }

  async function stop() {
    if (stopping) return;
    if (starting && !running) {
      generation += 1;
      await reset();
      return;
    }
    if (!running) return;
    const run = generation;
    const activeSocket = socket;
    stopping = true;
    setButton(true, true);
    setStatus("Dokončuji překlad…");
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (sourceNode) sourceNode.disconnect();
    if (captureNode) {
      // Port delivery can stall with a suspended worklet. Never leave a paid
      // Google session running just because the flush acknowledgement vanished.
      flushTimer = setTimeout(() => sendStop(run, activeSocket), 250);
      captureNode.port.postMessage({type: "flush"});
    } else {
      sendStop(run, activeSocket);
    }
  }

  toggle.addEventListener("click", () => {
    if (running || starting) void stop();
    else void start();
  });

  window.addEventListener("beforeunload", () => {
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  });
})();
