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
  let sessionEnded = false;
  let audioSent = false;
  let terminalErrorReceived = false;
  let speech = null;
  let primary = false;
  let fallbackPending = false;

  function setStatus(text, state = "idle") {
    status.textContent = text;
    status.dataset.state = state;
  }

  function setButton(active, disabled = false) {
    toggle.dataset.active = active ? "true" : "false";
    toggle.textContent = active ? (stopping ? "Dokončuji…" : "Zastavit") : "Spustit mikrofon";
    toggle.disabled = disabled;
  }

  function showError(code) {
    const messages = {
      audio_too_large: "Zvukový rámec je příliš velký.",
      audio_rate_exceeded: "Zvuk přicházel rychleji než v reálném čase.",
      invalid_audio: "Prohlížeč odeslal neplatný zvuk.",
      invalid_message: "Prohlížeč odeslal neplatný přepis.",
      transcript_too_large: "Přepis překročil bezpečnou délku. Spusťte novou relaci.",
      transcript_rate_exceeded: "Přepisy přicházely příliš rychle. Relace byla zastavena.",
      speech_denied: "Rozpoznávání nebo mikrofon nebyl povolen. Povolte jej v prohlížeči.",
      server_busy: "Server právě obsluhuje maximum živých relací.",
      google_rate_limited: "Google je právě vytížený. Zkuste to za chvíli.",
      google_timeout: "Google neodpověděl včas.",
      google_unavailable: "Google překlad je dočasně nedostupný.",
      interim_translation_failed: "Průběžný překlad je dočasně nedostupný. Finální překlad pokračuje.",
      translation_failed: "Jeden konečný překlad se nepodařilo vytvořit.",
      transcription_incomplete: "Závěr řeči nebyl potvrzen. Hotové překlady zůstávají; poslední část může chybět.",
      audio_backpressure: "Síť nestíhá přenášet zvuk. Relace byla zastavena; závěr může chybět.",
      translation_backlog_full: "Překlad nestíhá přijímanou řeč. Spusťte novou relaci.",
      session_limit: "Relace dosáhla časového limitu. Po dokončení překladů spusťte novou.",
      audio_unavailable: "Tento prohlížeč neumí bezpečně zpracovat mikrofon.",
      session_failed: "Relace byla přerušena. Zkuste ji spustit znovu."
    };
    errorShown = true;
    const message = messages[code] || "Překlad se nepodařilo dokončit.";
    if (code === "transcription_incomplete") {
      setStatus("Zastaveno · závěr nepotvrzen", "warning");
    } else {
      setStatus(message, "error");
    }
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
    speech?.dispose();
    speech = null;
    const closingSocket = socket;
    socket = null;
    stopping = true;
    setButton(false, true);
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
    if (!errorShown) setStatus(sessionEnded ? "Zastaveno" : "Připraveno");
    ui.endSession(status.textContent, status.dataset.state);
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

  function speechFailed(run, activeSocket, code) {
    if (run !== generation || socket !== activeSocket || stopping) return;
    const denied = ["not-allowed", "service-not-allowed"].includes(code);
    fallbackPending = !denied;
    stopping = true;
    if (denied) showError("speech_denied");
    else {
      errorShown = true;
      ui.showError("transcription_incomplete", "Web Speech vypadl. Přechod na Google zálohu může vynechat rozpracovanou řeč; přijaté finaly se dopřeloží.");
      setStatus("Web Speech nedostupný · dokončuji finaly a připojím Google zálohu…", "warning");
    }
    // Keep the Stop control usable during the drain so the user can cancel
    // automatic fallback. No audio is secretly recorded/queued in this gap.
    setButton(true, denied);
    if (!denied) toggle.textContent = "Zrušit přepnutí";
    void releaseAudio();
    sendStop(run, activeSocket);
  }

  function beginSpeech(run, activeSocket) {
    speech = window.createSpeechCapture({
      audioTrack: mediaStream.getAudioTracks()[0],
      onTranscript(type, text) {
        if (run !== generation || socket !== activeSocket || stopSent) return;
        if (activeSocket.bufferedAmount > MAX_BUFFERED_AUDIO_BYTES) {
          abortCongestedSession(run, activeSocket);
          return;
        }
        activeSocket.send(JSON.stringify({type, text}));
      },
      onFirstStt(value) {
        if (run === generation && socket === activeSocket) ui.telemetry({first_stt_ms: value});
      },
      onFailure(code) { speechFailed(run, activeSocket, code); },
      onStopped(confirmed) {
        if (run !== generation || socket !== activeSocket) return;
        if (!confirmed) showError("transcription_incomplete");
        void releaseAudio();
        sendStop(run, activeSocket);
      }
    });
    speech.start();
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
          if (primary) {
            // Meter and watchdog only: never send PCM to Google on this path.
            if (!stopping) {
              audioSent = true;
              ui.audioFrame(event.data);
              speech?.audioFrame(event.data);
            }
            return;
          }
          if (activeSocket.bufferedAmount + event.data.byteLength > MAX_BUFFERED_AUDIO_BYTES) {
            abortCongestedSession(run, activeSocket);
            return;
          }
          activeSocket.send(event.data);
          audioSent = true;
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
    if (primary) beginSpeech(run, activeSocket);
  }

  async function start(forceBackup = false) {
    if (starting || running || stopping) return;
    const run = ++generation;
    starting = true;
    errorShown = false;
    sessionEnded = false;
    audioSent = false;
    terminalErrorReceived = false;
    fallbackPending = false;
    primary = !forceBackup && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
    document.getElementById("stt-engine").textContent = primary ? "Web Speech" : "Google záloha";
    ui.beginSession();
    clearInterim();
    setButton(true, false);
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
      const activeSocket = new WebSocket(`${protocol}//${location.host}/ws/${primary ? "transcript" : "audio"}`);
      socket = activeSocket;
      activeSocket.binaryType = "arraybuffer";
      activeSocket.addEventListener("open", () => {
        if (run === generation && socket === activeSocket) {
          setStatus(primary ? "Připojuji překlad…" : "Připojuji Google zálohu…");
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
        } else if (message.type === "interim_reset") {
          if (!stopping) clearInterim();
        } else if (
          (message.type === "interim" || message.type === "final") &&
          typeof message.en === "string" && typeof message.ru === "string"
        ) {
          if (message.type === "interim" && stopping) return;
          if (message.type === "interim") renderInterim(message.en, message.ru);
          else renderFinal(message.en, message.ru);
          if (running && !stopping) {
            errorShown = false;
            setStatus("Poslouchám…", "live");
          }
        } else if (message.type === "telemetry") {
          ui.telemetry(message);
        } else if (message.type === "error") {
          if (message.code === "interim_translation_failed" && stopping) return;
          showError(message.code);
          if (!["translation_failed", "interim_translation_failed"].includes(message.code)) {
            fallbackPending = false;
            speech?.dispose();
            if (primary && message.code === "session_limit" && audioSent) {
              ui.showError("transcription_incomplete", "Časový limit přerušil Web Speech. Rozpracovaný závěr nemusí být úplný.");
            }
            terminalErrorReceived = message.code !== "session_limit";
            stopping = true;
            stopSent = true;
            clearTimeout(flushTimer);
            flushTimer = null;
            setButton(running, true);
            await releaseAudio();
          }
        } else if (message.type === "ended") {
          sessionEnded = true;
          if (!errorShown) setStatus("Zastaveno");
        }
      });
      activeSocket.addEventListener("error", () => {
        if (run === generation && socket === activeSocket) {
          fallbackPending = false;
          showError("session_failed");
        }
      });
      activeSocket.addEventListener("close", async () => {
        if (run !== generation || socket !== activeSocket) return;
        // A transport close alone does not confirm that Stop drained the tail.
        if (!sessionEnded && !terminalErrorReceived) {
          if (stopping && audioSent) showError("transcription_incomplete");
          else if (!stopping && !errorShown) showError("session_failed");
        }
        const useBackup = fallbackPending && sessionEnded;
        fallbackPending = false;
        await reset(false);
        // reset increments exactly once; a newer user session supersedes this.
        if (useBackup && generation === run + 1) await start(true);
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
    if (fallbackPending) {
      fallbackPending = false;
      setButton(true, true);
      setStatus("Dokončuji překlad…");
      return;
    }
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
    ui.stopping();
    if (primary) {
      // Recognition owns its microphone. Give stop() its final-result chance
      // before closing either the recognizer or the translation WebSocket.
      speech?.stop();
      return;
    }
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
    fallbackPending = false;
    speech?.dispose();
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  });
})();
