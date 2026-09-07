// Run the actual presentation and audio modules against deterministic browser APIs.
// No microphone, server, credentials, or paid provider requests are used.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

const scripts = ['interim-layout.js', 'translator-ui.js', 'web-speech.js', 'audio-session.js'].map(name => ({
  name, source: readFileSync(join(__dirname, '../app/static', name), 'utf8')
}));
const settle = () => new Promise(resolve => setImmediate(resolve));

function harness({saved = null, storageBlocked = false, fullscreenFails = false, webSpeech = false} = {}) {
  class Element {
    constructor() {
      this.dataset = {};
      this.children = [];
      this.textContent = '';
      this.hidden = false;
      this.disabled = false;
      this.listeners = {};
      this.style = {setProperty(name, value) { this[name] = value; }};
      this.scrollTop = 0;
      this.scrollHeight = 600;
      this.clientHeight = 300;
    }
    addEventListener(type, listener) {
      (this.listeners[type] ||= []).push(listener);
    }
    setAttribute(name, value) { this[name] = value; }
    getBoundingClientRect() { return this.rect || {top: 200, bottom: 800}; }
    append(value) { this.children.push(value); }
    prepend(value) { this.children.unshift(value); this.onPrepend?.(value); }
    async dispatch(type, event = {}) {
      for (const listener of this.listeners[type] || []) await listener(event);
      await settle();
    }
  }

  const ids = Object.fromEntries([
    'toggle', 'status', 'workspace', 'live-heading', 'live-panel', 'live-en', 'live-ru', 'history', 'result-template',
    'history-scroll', 'history-empty', 'history-count', 'live-note', 'font-size',
    'font-down', 'font-up', 'font-value', 'follow-latest', 'fullscreen-toggle',
    'compact-toggle', 'theme-toggle', 'toolbar', 'presentation-notice',
    'stat-audio', 'stat-interim', 'stat-final', 'stat-stt', 'stat-translation',
    'stat-queue', 'stat-hypotheses', 'stat-connect', 'audio-level', 'stt-engine'
  ].map(id => [id, new Element()]));
  ids['result-template'].content = {
    cloneNode() {
      const en = new Element();
      const ru = new Element();
      return {en, ru, querySelector: selector => ({'.result-en': en, '.result-ru': ru})[selector]};
    }
  };
  const sockets = [];
  const worklets = [];
  const contexts = [];
  const tracks = [];
  const timers = new Map();
  const frames = new Map();
  const windowEvents = new Element();
  const visualViewport = new Element();
  visualViewport.height = 900;
  visualViewport.offsetTop = 0;
  const observers = [];
  let nextTimer = 0;
  let now = 1234;
  const recognitions = [];
  class Recognition {
    constructor() {
      if (webSpeech === 'constructor-failure') throw new Error('unavailable');
      recognitions.push(this);
    }
    start(track) { this.started = true; this.track = track; }
    stop() { this.stopped = true; }
    abort() { this.aborted = true; }
    result(items, resultIndex = 0) {
      this.onresult?.({resultIndex, results: items.map(([text, isFinal]) =>
        Object.assign([{transcript: text}], {isFinal}))});
    }
    error(error) { this.onerror?.({error}); }
    end() { this.onend?.(); }
  }

  class Socket extends Element {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
    constructor(url) {
      super();
      this.url = url;
      this.readyState = Socket.CONNECTING;
      this.bufferedAmount = 0;
      this.sent = [];
      sockets.push(this);
    }
    send(value) { this.sent.push(value); }
    close(code = 1000) {
      this.closeCode = code;
      this.readyState = Socket.CLOSED;
      void this.dispatch('close', {code});
    }
    async open() {
      this.readyState = Socket.OPEN;
      await this.dispatch('open');
      await this.message({type: 'ready'});
    }
    message(value) { return this.dispatch('message', {data: JSON.stringify(value)}); }
  }

  const node = () => ({connect() {}, disconnect() { this.disconnected = true; }});
  class Context {
    constructor() {
      this.state = 'suspended';
      this.audioWorklet = {addModule: async () => {}};
      this.destination = {};
      contexts.push(this);
    }
    async resume() { this.state = 'running'; }
    async close() { this.state = 'closed'; }
    createMediaStreamSource() { return node(); }
    createGain() { return {...node(), gain: {value: 1}}; }
  }
  class Worklet {
    constructor() {
      Object.assign(this, node());
      this.port = {
        onmessage: null,
        sent: [],
        postMessage(value) { this.sent.push(value); },
        close() { this.closed = true; }
      };
      worklets.push(this);
    }
  }
  const document = new Element();
  const root = new Element();
  document.documentElement = root;
  document.getElementById = id => ids[id];
  document.createElement = () => new Element();
  document.fullscreenEnabled = true;
  root.requestFullscreen = async () => {
    if (fullscreenFails) throw new Error('not allowed');
    document.fullscreenElement = root;
    await document.dispatch('fullscreenchange');
  };
  document.exitFullscreen = async () => {
    document.fullscreenElement = null;
    await document.dispatch('fullscreenchange');
  };
  const storage = new Map(saved === null ? [] : [['translator.presentation.v1', saved]]);
  const sandbox = {
    document,
    localStorage: {
      getItem(key) { if (storageBlocked) throw new Error('blocked'); return storage.get(key) ?? null; },
      setItem(key, value) { if (storageBlocked) throw new Error('blocked'); storage.set(key, value); }
    },
    performance: {now: () => now},
    ...(webSpeech ? {webkitSpeechRecognition: Recognition} : {}),
    matchMedia: () => ({matches: false, addEventListener() {}}),
    location: {protocol: 'https:', host: 'translator.test'},
    navigator: {mediaDevices: {getUserMedia: async () => {
      const track = {stopped: false, stop() { this.stopped = true; }};
      tracks.push(track);
      return {getTracks: () => [track], getAudioTracks: () => [track]};
    }}},
    WebSocket: Socket, AudioContext: Context, AudioWorkletNode: Worklet,
    ArrayBuffer, DataView, console,
    innerHeight: 900, visualViewport,
    getComputedStyle: () => ({rowGap: '4px'}),
    requestAnimationFrame(callback) { frames.set(++nextTimer, callback); return nextTimer; },
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); }
      observe(target) { this.targets.push(target); }
    },
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    addEventListener(type, listener) { windowEvents.addEventListener(type, listener); }
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const script of scripts) vm.runInContext(script.source, sandbox, {filename: script.name});

  return {
    ids, sockets, worklets, contexts, tracks, timers, root, document, storage, recognitions,
    frames, windowEvents, visualViewport, observers,
    paint() {
      const pending = [...frames.values()]; frames.clear();
      for (const callback of pending) callback();
    },
    advance(ms) { now += ms; },
    async click() {
      assert.equal(ids.toggle.disabled, false);
      await ids.toggle.dispatch('click');
    },
    async start() {
      await this.click();
      const socket = sockets.at(-1);
      await socket.open();
      assert.equal(ids.status.textContent, 'Poslouchám…');
      return socket;
    },
    async port(value, worklet = worklets.at(-1)) {
      worklet.port.onmessage?.({data: value});
      await settle();
    }
  };
}

function previewGeometry(h, {width = 480, lineHeight = 1.35} = {}) {
  const font = () => parseFloat(h.ids['live-panel'].style['--live-font-size']) || 28;
  for (const id of ['live-en', 'live-ru']) {
    const text = h.ids[id];
    Object.defineProperties(text, {
      clientWidth: {get: () => width},
      scrollWidth: {get: () => width},
      clientHeight: {get: () => Math.max(0, parseFloat(h.ids['live-panel'].style['--live-max-height']) - 48)},
      scrollHeight: {get: () => Math.ceil(text.textContent.length * font() * 0.55 / width) * font() * lineHeight}
    });
  }
}

test('preview uses available height before shrinking and fits both languages together', async () => {
  const h = harness();
  previewGeometry(h);
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'word '.repeat(55), ru: 'слово '.repeat(65)});
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-max-height'], '548px');
  // 492 px of RU content, more than the old 28vh limit, fits at the chosen size.
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '28px');
  await socket.message({type: 'interim', en: 'word '.repeat(55), ru: 'слово '.repeat(180)});
  assert.equal(h.frames.size, 1);
  h.paint();
  const fitted = parseFloat(h.ids['live-panel'].style['--live-font-size']);
  assert.ok(fitted >= 14 && fitted < 28);
  assert.ok(h.ids['live-en'].scrollHeight <= h.ids['live-en'].clientHeight + 1);
  assert.ok(h.ids['live-ru'].scrollHeight <= h.ids['live-ru'].clientHeight + 1);
  assert.equal(h.root.style['--translation-size'], '28px');
  assert.equal(h.ids['font-value'].textContent, '28 px');
  assert.equal(h.storage.size, 0);
});

test('preview fitting follows viewport, toolbar and fullscreen geometry changes', async () => {
  const h = harness();
  previewGeometry(h);
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'word '.repeat(160), ru: 'слово '.repeat(160)});
  h.paint();
  const initial = parseFloat(h.ids['live-panel'].style['--live-font-size']);
  h.visualViewport.height = 600;
  await h.visualViewport.dispatch('resize');
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-max-height'], '348px');
  assert.ok(parseFloat(h.ids['live-panel'].style['--live-font-size']) < initial);
  h.visualViewport.height = 1600;
  h.ids.workspace.rect = {top: 100, bottom: 1600};
  h.ids['live-panel'].rect = {top: 120};
  h.observers[0].callback();
  await h.document.dispatch('fullscreenchange');
  assert.equal(h.frames.size, 1);
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '28px');
  assert.deepEqual(h.observers[0].targets, [h.ids.workspace, h.ids['live-heading']]);
  assert.equal(h.frames.size, 0); // No fit/resize loop caused by observing the preview itself.
});

test('shorter draft, final and reset restore font without resizing final history', async () => {
  const h = harness();
  previewGeometry(h);
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'long '.repeat(300), ru: 'долго '.repeat(300)});
  h.paint();
  assert.ok(parseFloat(h.ids['live-panel'].style['--live-font-size']) < 28);
  h.ids['font-size'].value = '40';
  await h.ids['font-size'].dispatch('input');
  h.paint();
  assert.equal(h.root.style['--translation-size'], '40px');
  assert.ok(parseFloat(h.ids['live-panel'].style['--live-font-size']) < 40);
  await socket.message({type: 'interim', en: 'short', ru: 'коротко'});
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '40px');
  await socket.message({type: 'interim', en: 'long '.repeat(300), ru: 'долго '.repeat(300)});
  h.paint();
  await socket.message({type: 'final', en: 'saved', ru: 'сохранено'});
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '40px');
  h.paint();
  assert.equal(h.ids.history.children[0].en.style['--live-font-size'], undefined);
  assert.equal(h.ids['live-en'].scrollTop, 0);
  assert.deepEqual(Object.keys(JSON.parse(h.storage.get('translator.presentation.v1'))).sort(),
    ['compact', 'follow', 'font', 'theme']);
});

test('readable font floor retains all text and respects paused following', async () => {
  const h = harness();
  previewGeometry(h, {width: 160});
  const socket = await h.start();
  h.ids['follow-latest'].checked = false;
  await h.ids['follow-latest'].dispatch('change');
  h.ids['live-en'].scrollTop = 50;
  h.ids['live-ru'].scrollTop = 90;
  const en = 'long '.repeat(2400), ru = 'длинно '.repeat(1700);
  await socket.message({type: 'interim', en, ru});
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '14px');
  assert.equal(h.ids['live-en'].textContent, en);
  assert.equal(h.ids['live-ru'].textContent, ru);
  assert.equal(h.ids['live-en'].scrollTop, 50);
  assert.equal(h.ids['live-ru'].scrollTop, 90);
  assert.ok(h.ids['live-ru'].scrollHeight > h.ids['live-ru'].clientHeight);
  await socket.message({type: 'interim_reset'});
  h.paint();
  assert.equal(h.ids['live-panel'].style['--live-font-size'], '28px');
});

test('Web Speech is primary, captures locally and sends only Czech transcript JSON', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  assert.match(socket.url, /\/ws\/transcript$/);
  assert.equal(h.ids['stt-engine'].textContent, 'Web Speech');
  const speech = h.recognitions[0];
  assert.equal(speech.lang, 'cs-CZ');
  assert.equal(speech.continuous, true);
  assert.equal(speech.interimResults, true);
  assert.equal(speech.maxAlternatives, 1);
  assert.equal(speech.track, h.tracks[0]);
  await h.port(new Int16Array(1600).fill(8000).buffer);
  assert.equal(socket.sent.length, 0);
  assert.ok(h.ids['audio-level'].value > 0);
  h.advance(210);
  speech.result([['dobrý', false]]);
  speech.result([['dobrý den', false]]);
  assert.deepEqual(socket.sent.map(JSON.parse), [
    {type: 'interim', text: 'dobrý'}, {type: 'interim', text: 'dobrý den'}
  ]);
  assert.equal(h.ids['stat-stt'].textContent, '210 ms');
  await socket.message({type: 'interim', en: 'hello', ru: 'здравствуйте'});
  assert.equal(h.ids['live-en'].textContent, 'hello');
  assert.equal(h.ids['live-ru'].textContent, 'здравствуйте');
});

test('resultIndex avoids replaying finals but preserves repeated words in distinct finals', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  const speech = h.recognitions[0];
  speech.result([['ano', true], ['zítra', false]]);
  speech.result([['ano', true], ['ano', true], ['dnes', false]], 1);
  speech.result([['ano', true], ['ano', true], ['dnes', false]], 0);
  assert.deepEqual(socket.sent.map(JSON.parse).filter(x => x.type === 'final'),
    [{type: 'final', text: 'ano'}, {type: 'final', text: 'ano'}]);
  speech.result([['ano', true], ['ano', true]], 2);
  assert.deepEqual(JSON.parse(socket.sent.at(-1)), {type: 'interim', text: ''});
  await socket.message({type: 'interim', en: 'old', ru: 'старое'});
  await socket.message({type: 'interim_reset'});
  assert.equal(h.ids['live-en'].textContent, '');
});

test('Web Speech Stop accepts late final before sending Stop, without fake completion', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  const speech = h.recognitions[0];
  speech.result([['rozpracované', false]]);
  await h.click();
  assert.equal(speech.stopped, true);
  assert.equal(speech.aborted, undefined);
  assert.equal(socket.sent.some(x => JSON.parse(x).type === 'stop'), false);
  speech.result([['Dokončená věta.', true]]);
  speech.end();
  await settle();
  assert.deepEqual(socket.sent.map(JSON.parse).slice(-2), [
    {type: 'final', text: 'Dokončená věta.'}, {type: 'stop'}
  ]);
  assert.equal(h.tracks[0].stopped, true);
  await socket.message({type: 'final', en: 'Done.', ru: 'Готово.'});
  await socket.message({type: 'ended'});
  socket.close();
  await settle();
  assert.equal(h.ids.status.textContent, 'Zastaveno');
  assert.equal(h.ids.history.children.length, 1);
});

test('Web Speech Stop timeout or remaining hypothesis warns and never invents a final', async () => {
  for (const timeout of [true, false]) {
    const h = harness({webSpeech: true});
    const socket = await h.start();
    const speech = h.recognitions[0];
    speech.result([['nedokončené', false]]);
    await socket.message({type: 'interim', en: 'unfinished', ru: 'не закончено'});
    await h.click();
    if (timeout) [...h.timers.values()][0]();
    else speech.end();
    await socket.message({type: 'ended'});
    socket.close();
    await settle();
    assert.equal(socket.sent.map(JSON.parse).filter(x => x.type === 'final').length, 0);
    assert.equal(h.ids.history.children[0].dataset.code, 'transcription_incomplete');
    assert.equal(h.ids.history.children[0].children.at(-1).en.textContent, 'unfinished');
    assert.equal(h.sockets.length, 1);
  }
});

test('service failure drains accepted finals before Google fallback and ignores old callbacks', async () => {
  const h = harness({webSpeech: true});
  const primary = await h.start();
  const speech = h.recognitions[0];
  const oldResult = speech.onresult;
  speech.result([['Hotovo.', true], ['další', false]]);
  speech.error('network');
  await settle();
  assert.equal(speech.aborted, true);
  assert.equal(h.tracks[0].stopped, true);
  assert.equal(h.sockets.length, 1);
  assert.equal(primary.sent.at(-1), '{"type":"stop"}');
  await primary.message({type: 'final', en: 'Done.', ru: 'Готово.'});
  await primary.message({type: 'ended'});
  primary.close();
  await settle();
  await settle();
  const backup = h.sockets[1];
  assert.match(backup.url, /\/ws\/audio$/);
  await backup.open();
  assert.equal(h.ids['stt-engine'].textContent, 'Google záloha');
  oldResult({resultIndex: 0, results: [Object.assign([{transcript: 'obsolete'}], {isFinal: true})]});
  await h.port(new ArrayBuffer(3200));
  assert.equal(backup.sent.length, 1);
  assert.equal(backup.sent[0].byteLength, 3200);
  assert.equal(h.ids.history.children[0].en.textContent, 'Done.');
  assert.equal(h.recognitions.length, 1);
});

test('unavailable Web Speech constructor or capture service can use the backup', async () => {
  for (const brokenConstructor of [true, false]) {
    const h = harness({webSpeech: brokenConstructor ? 'constructor-failure' : true});
    await h.click();
    const socket = h.sockets[0];
    await socket.open();
    if (!brokenConstructor) h.recognitions[0].error('audio-capture');
    await socket.message({type: 'ended'});
    socket.close();
    await settle();
    await settle();
    assert.match(h.sockets[1].url, /\/ws\/audio$/);
  }
});

test('Stop can cancel startup before ready without resurrecting capture', async () => {
  const h = harness({webSpeech: true});
  await h.click();
  const oldSocket = h.sockets[0];
  await h.click();
  await oldSocket.message({type: 'ready'});
  assert.equal(h.recognitions.length, 0);
  assert.equal(h.worklets.length, 0);
  assert.equal(h.tracks[0].stopped, true);
  assert.equal(h.ids.toggle.dataset.active, 'false');
});

test('Stop during fallback drain prevents automatic restart', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  h.recognitions[0].error('network');
  await h.click();
  await socket.message({type: 'ended'});
  socket.close();
  await settle();
  assert.equal(h.sockets.length, 1);
  assert.equal(h.ids.toggle.disabled, false);
});

test('permission denial never activates a different STT service', async () => {
  for (const error of ['not-allowed', 'service-not-allowed']) {
    const h = harness({webSpeech: true});
    const socket = await h.start();
    h.recognitions[0].error(error);
    await socket.message({type: 'ended'});
    socket.close();
    await settle();
    assert.equal(h.sockets.length, 1);
    assert.match(h.ids.status.textContent, /nebyl povolen/);
    assert.equal(h.tracks[0].stopped, true);
  }
});

test('translation/backend failure never triggers Google STT fallback', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  await socket.message({type: 'error', code: 'interim_translation_failed'});
  assert.equal(h.recognitions[0].aborted, undefined);
  await socket.message({type: 'error', code: 'google_unavailable'});
  assert.equal(h.recognitions[0].aborted, true);
  socket.close();
  await settle();
  assert.equal(h.sockets.length, 1);
});

test('quiet auto-end restarts Web Speech with fresh result indices, without Google audio', async () => {
  const h = harness({webSpeech: true});
  const socket = await h.start();
  h.advance(5000);
  h.recognitions[0].error('no-speech');
  h.recognitions[0].end();
  [...h.timers.values()][0]();
  h.recognitions[1].result([['ano', true]]);
  assert.equal(h.recognitions.length, 2);
  assert.equal(h.sockets.length, 1);
  assert.deepEqual(socket.sent.map(JSON.parse), [{type: 'final', text: 'ano'}]);
});

test('watchdog ignores silence but falls back on sustained audio without results', async () => {
  const h = harness({webSpeech: true});
  await h.start();
  for (let i = 0; i < 120; i++) {
    h.advance(100);
    await h.port(new ArrayBuffer(3200));
  }
  assert.equal(h.recognitions[0].aborted, undefined);
  for (let i = 0; i < 60; i++) {
    h.advance(100);
    await h.port(new Int16Array(1600).fill(5000).buffer);
  }
  assert.equal(h.recognitions[0].aborted, true);
  assert.match(h.ids.toggle.textContent, /Zrušit přepnutí/);
});

test('old audio, flush and fallback callbacks cannot affect a restarted session', async () => {
  const h = harness();
  const first = await h.start();
  const oldWorklet = h.worklets[0];
  const oldHandler = oldWorklet.port.onmessage;
  await h.click();
  const oldFallback = [...h.timers.values()][0];
  oldFallback(); // missing flush acknowledgement -> normal fallback Stop
  assert.equal(first.sent.at(-1), '{"type":"stop"}');
  first.close();
  await settle();
  assert.equal(oldWorklet.port.onmessage, null);
  assert.equal(oldWorklet.port.closed, true);
  const second = await h.start();
  oldHandler({data: new ArrayBuffer(3200)});
  oldHandler({data: {type: 'flushed'}});
  oldFallback();
  await settle();
  assert.equal(second.sent.length, 0);
  assert.equal(h.ids.status.textContent, 'Poslouchám…');
  await h.port(new ArrayBuffer(3200));
  assert.equal(second.sent.length, 1);
  await h.click();
  await h.port({type: 'flushed'});
  assert.equal(second.sent.at(-1), '{"type":"stop"}');
  assert.equal(first.sent.length, 1);
});

test('normal Stop sends the partial PCM tail before exactly one stop message', async () => {
  const h = harness();
  const socket = await h.start();
  await h.port(new ArrayBuffer(3200));
  await h.click();
  const fallback = [...h.timers.values()][0];
  await h.port(new ArrayBuffer(200));
  await h.port({type: 'flushed'});
  fallback();
  assert.deepEqual(socket.sent.map(x => typeof x === 'string' ? x : x.byteLength),
    [3200, 200, '{"type":"stop"}']);
});

test('failed final leaves a permanent gap despite later successful translations', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'draft', ru: 'черновик'});
  await socket.message({type: 'error', code: 'translation_failed', recoverable: true});
  await socket.message({type: 'interim', en: 'next', ru: 'дальше'});
  assert.equal(h.ids.status.textContent, 'Poslouchám…');
  const gap = h.ids.history.children[0];
  assert.equal(gap.dataset.code, 'translation_failed');
  assert.match(gap.textContent, /nepodařilo/);
  await socket.message({type: 'final', en: 'Done', ru: 'Готово'});
  assert.equal(h.ids.history.children[1], gap);
  assert.equal(h.ids.history.children[0].en.textContent, 'Done');
  assert.equal(h.ids.history.children[0].ru.textContent, 'Готово');
});

test('reset clears speculative text while preserving final history and gaps', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'final', en: 'saved', ru: 'сохранено'});
  await socket.message({type: 'interim', en: 'old', ru: 'старое'});
  socket.close();
  await settle();
  assert.equal(h.ids['live-panel'].hidden, false);
  assert.equal(h.ids['live-en'].textContent, '');
  assert.equal(h.ids['live-ru'].textContent, '');
  await h.start();
  assert.equal(h.ids['live-panel'].hidden, false);
  assert.equal(h.ids.history.children[0].en.textContent, 'saved');
});

test('network backlog aborts capture visibly without silently skipping an audio frame', async () => {
  const h = harness();
  const socket = await h.start();
  socket.bufferedAmount = 320000;
  await h.port(new ArrayBuffer(3200));
  assert.equal(socket.sent.length, 0);
  assert.equal(socket.closeCode, 4000);
  assert.equal(h.tracks[0].stopped, true);
  assert.equal(h.contexts[0].state, 'closed');
  assert.equal(h.worklets[0].port.closed, true);
  assert.equal(h.ids.history.children[0].dataset.code, 'audio_backpressure');
  assert.match(h.ids.status.textContent, /Síť/);
  assert.equal(h.ids.toggle.disabled, false);
});

test('interim failure is visible without stopping audio or deleting the last paired preview', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'last preview', ru: 'последний черновик'});
  await socket.message({type: 'error', code: 'interim_translation_failed', recoverable: true});
  assert.equal(h.ids['live-panel'].dataset.state, 'error');
  assert.equal(h.ids['live-en'].textContent, 'last preview');
  assert.equal(h.ids['live-ru'].textContent, 'последний черновик');
  assert.equal(h.tracks[0].stopped, false);
  assert.equal(socket.readyState, 1);
  assert.equal(h.ids.history.children.length, 0);
  await socket.message({type: 'telemetry', kind: 'interim', stage: 'translating'});
  assert.equal(h.ids['live-panel'].dataset.state, 'error');
  await socket.message({type: 'interim', en: 'recovered', ru: 'восстановлено'});
  assert.equal(h.ids['live-panel'].dataset.state, 'live');
  assert.equal(h.ids['stat-interim'].textContent, '2');
  assert.equal(h.ids.status.dataset.state, 'live');
});

test('telemetry updates measured timings without fabricating missing values or changing translations', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'interim', en: 'hello', ru: 'привет'});
  await socket.message({type: 'telemetry', kind: 'interim', stage: 'translated', first_stt_ms: 310,
    translation_ms: 450, queue_ms: 350, hypotheses: 12, finals_received: 0});
  assert.equal(h.ids['stat-stt'].textContent, '310 ms');
  assert.equal(h.ids['stat-translation'].textContent, '450 ms');
  assert.equal(h.ids['stat-queue'].textContent, '350 ms');
  assert.equal(h.ids['stat-hypotheses'].textContent, '12 / 0');
  await socket.message({type: 'telemetry', stage: 'translated', first_stt_ms: null, translation_ms: 'bad', queue_ms: -1});
  assert.equal(h.ids['stat-stt'].textContent, '310 ms');
  assert.equal(h.ids['stat-translation'].textContent, '450 ms');
  assert.equal(h.ids['stat-queue'].textContent, '350 ms');
  assert.equal(h.ids['live-en'].textContent, 'hello');
  await h.port(new Int16Array(1600).fill(8000).buffer);
  assert.ok(h.ids['audio-level'].value > 0);
  socket.close();
  await settle();
  assert.equal(h.ids['audio-level'].value, 0);
});

test('presentation preferences are bounded, persisted and cannot configure the audio session', async () => {
  const h = harness({saved: JSON.stringify({font: 999, compact: false, follow: true, theme: 'bad', engine: 'legacy'})});
  assert.equal(h.root.style['--translation-size'], '64px');
  assert.equal(h.ids['font-up'].disabled, true);
  await h.ids['font-down'].dispatch('click');
  assert.equal(h.root.style['--translation-size'], '62px');
  h.ids['font-size'].value = '18';
  await h.ids['font-size'].dispatch('input');
  assert.equal(h.ids['font-down'].disabled, true);
  await h.ids['compact-toggle'].dispatch('click');
  assert.equal(h.ids.toolbar.dataset.compact, 'true');
  assert.equal(h.ids['compact-toggle']['aria-expanded'], 'false');
  await h.ids['theme-toggle'].dispatch('click');
  assert.equal(h.root.dataset.theme, 'dark');
  const prefs = JSON.parse(h.storage.get('translator.presentation.v1'));
  assert.deepEqual(Object.keys(prefs).sort(), ['compact', 'follow', 'font', 'theme']);
  const socket = await h.start();
  assert.equal(socket.sent.length, 0); // No preference/config control messages.
});

test('broken or blocked local storage cannot prevent startup and presentation controls', async () => {
  for (const options of [{saved: 'not-json'}, {storageBlocked: true}]) {
    const h = harness(options);
    assert.equal(h.root.style['--translation-size'], '28px');
    await h.ids['font-up'].dispatch('click');
    assert.equal(h.root.style['--translation-size'], '30px');
    await h.start();
  }
});

test('fullscreen responds to Escape and permission errors without disturbing capture', async () => {
  const h = harness();
  await h.start();
  await h.ids['fullscreen-toggle'].dispatch('click');
  assert.equal(h.ids['fullscreen-toggle']['aria-pressed'], 'true');
  await h.document.exitFullscreen();
  assert.equal(h.ids['fullscreen-toggle']['aria-pressed'], 'false');
  assert.equal(h.tracks[0].stopped, false);
  const blocked = harness({fullscreenFails: true});
  await blocked.ids['fullscreen-toggle'].dispatch('click');
  assert.equal(blocked.ids['fullscreen-toggle']['aria-pressed'], 'false');
  assert.match(blocked.ids['presentation-notice'].textContent, /nepovolil/);
});

test('newest-first history preserves the reading position while following is paused', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'final', en: 'first', ru: 'первый'});
  assert.equal(h.ids['history-scroll'].scrollTop, 0);
  h.ids['history-scroll'].scrollTop = 120;
  await h.ids['history-scroll'].dispatch('scroll');
  assert.equal(h.ids['follow-latest'].checked, false);
  h.ids.history.onPrepend = () => { h.ids['history-scroll'].scrollHeight += 180; };
  await socket.message({type: 'final', en: 'second', ru: 'второй'});
  assert.equal(h.ids['history-scroll'].scrollTop, 300);
  assert.equal(h.ids.history.children[0].en.textContent, 'second');
  assert.equal(h.ids.history.children[1].en.textContent, 'first');
  await socket.message({type: 'error', code: 'translation_failed'});
  assert.equal(h.ids['history-scroll'].scrollTop, 480);
  await socket.message({type: 'interim', en: '<img onerror=bad>', ru: '<script>bad</script>'});
  assert.equal(h.ids['live-en'].textContent, '<img onerror=bad>');
  assert.equal(h.ids['live-ru'].textContent, '<script>bad</script>');
  assert.equal(h.ids['live-panel'].hidden, false);
  h.ids['follow-latest'].checked = true;
  await h.ids['follow-latest'].dispatch('change');
  assert.equal(h.ids['history-scroll'].scrollTop, 0);
  await h.ids['history-scroll'].dispatch('scroll');
  assert.equal(h.ids['follow-latest'].checked, true);
});

test('session stats reset on restart but final history and error warnings survive', async () => {
  const h = harness();
  const socket = await h.start();
  for (let frame = 0; frame < 10; frame++) await h.port(new ArrayBuffer(3200));
  assert.equal(h.ids['stat-audio'].textContent, '00:01');
  await socket.message({type: 'final', en: 'saved', ru: 'сохранено'});
  await socket.message({type: 'error', code: 'translation_failed'});
  socket.close();
  await settle();
  const next = await h.start();
  assert.equal(h.ids['stat-audio'].textContent, '00:00');
  assert.equal(h.ids['stat-final'].textContent, '0');
  assert.equal(h.ids.history.children.length, 2);
  await next.message({type: 'final', en: 'next', ru: 'дальше'});
  assert.equal(h.ids['history-count'].textContent, '2');
  assert.equal(h.ids.history.children[0].en.textContent, 'next');
  assert.equal(h.ids.history.children[1].dataset.code, 'translation_failed');
  assert.equal(h.ids.history.children[2].en.textContent, 'saved');
});

test('terminal error releases audio immediately and final cannot restore live status', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'error', code: 'transcription_incomplete'});
  await socket.message({type: 'final', en: 'last accepted', ru: 'принято'});
  assert.equal(h.tracks[0].stopped, true);
  assert.equal(h.worklets[0].port.closed, true);
  assert.match(h.ids.status.textContent, /závěr nepotvrzen/);
  assert.equal(h.ids.status.dataset.state, 'warning');
  assert.equal(h.ids.history.children[1].dataset.code, 'transcription_incomplete');
});

test('complete final pairs stay newest first beneath a separate interim', async () => {
  const h = harness();
  const socket = await h.start();
  const en = 'A complete final paragraph. '.repeat(80);
  const ru = 'Полный окончательный перевод. '.repeat(80);
  await socket.message({type: 'final', en: 'older', ru: 'раньше'});
  await socket.message({type: 'final', en, ru});
  await socket.message({type: 'interim', en: 'live', ru: 'сейчас'});
  assert.equal(h.ids.history.children.length, 2);
  assert.equal(h.ids.history.children[0].en.textContent, en);
  assert.equal(h.ids.history.children[0].ru.textContent, ru);
  assert.equal(h.ids.history.children[1].en.textContent, 'older');
  assert.equal(h.ids['live-en'].textContent, 'live');
  assert.equal(h.ids['live-ru'].textContent, 'сейчас');
  assert.equal(h.ids['history-scroll'].scrollTop, 0);
});

test('Stop shows finishing, releases the microphone and keeps successful completion after close', async () => {
  const h = harness();
  const socket = await h.start();
  await h.port(new ArrayBuffer(3200));
  await socket.message({type: 'interim', en: 'draft', ru: 'черновик'});
  await h.click();
  assert.equal(h.ids.toggle.disabled, true);
  assert.equal(h.ids.toggle.textContent, 'Dokončuji…');
  assert.match(h.ids['live-note'].textContent, /Dokončuji/);
  assert.equal(h.tracks[0].stopped, true);
  await h.port({type: 'flushed'});
  await socket.message({type: 'interim', en: 'late draft', ru: 'поздно'});
  assert.equal(h.ids['live-en'].textContent, 'draft');
  await socket.message({type: 'error', code: 'interim_translation_failed'});
  assert.match(h.ids['live-note'].textContent, /Dokončuji/);
  await socket.message({type: 'final', en: 'complete', ru: 'готово'});
  await socket.message({type: 'ended'});
  socket.close();
  await settle();
  assert.equal(h.ids.toggle.disabled, false);
  assert.equal(h.ids.status.textContent, 'Zastaveno');
  assert.equal(h.ids['live-note'].textContent, 'Zastaveno');
  assert.equal(h.ids['live-panel'].dataset.active, 'false');
  assert.equal(h.ids.history.children.length, 1);
  assert.equal(h.ids.history.children[0].en.textContent, 'complete');
});

test('empty Start/Stop leaves no history warning and can restart', async () => {
  const h = harness();
  const socket = await h.start();
  await h.click();
  await h.port({type: 'flushed'});
  await socket.message({type: 'ended'});
  socket.close();
  await settle();
  assert.equal(h.ids.status.textContent, 'Zastaveno');
  assert.equal(h.ids.history.children.length, 0);
  await h.start();
  assert.equal(h.ids['live-panel'].dataset.active, 'true');
});

test('unconfirmed Stop warns once per session without hiding distinct failed finals', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'final', en: 'saved', ru: 'сохранено'});
  await h.port(new ArrayBuffer(200));
  await h.click();
  await h.port({type: 'flushed'});
  for (let index = 0; index < 2; index++) {
    await socket.message({type: 'error', code: 'transcription_incomplete', recoverable: true});
  }
  assert.equal(h.ids.history.children.length, 2);
  assert.equal(h.ids.history.children[0].className, 'result result-warning');
  await socket.message({type: 'final', en: 'last accepted', ru: 'последнее'});
  await socket.message({type: 'telemetry', kind: 'final', stage: 'translating'});
  socket.close(1011);
  await settle();
  assert.equal(h.ids.history.children.length, 3);
  assert.equal(h.ids.status.dataset.state, 'warning');
  assert.equal(h.ids['live-panel'].dataset.state, 'warning');
  assert.equal(h.ids['live-panel'].dataset.active, 'false');
  assert.match(h.ids['live-note'].textContent, /závěr nepotvrzen/);
  const next = await h.start();
  await next.message({type: 'error', code: 'translation_failed'});
  await next.message({type: 'error', code: 'translation_failed'});
  assert.equal(h.ids.history.children.length, 5);
  await next.message({type: 'error', code: 'transcription_incomplete'});
  assert.equal(h.ids.history.children.length, 6);
});

test('close without ended cannot silently claim a successful Stop after subsecond audio', async () => {
  const h = harness();
  const socket = await h.start();
  await h.port(new ArrayBuffer(200));
  assert.equal(h.ids['stat-audio'].textContent, '00:00');
  await socket.message({type: 'error', code: 'interim_translation_failed'});
  await h.click();
  await h.port({type: 'flushed'});
  socket.close();
  await settle();
  assert.equal(h.ids.history.children[0].dataset.code, 'transcription_incomplete');
  assert.equal(h.ids.status.dataset.state, 'warning');
});

test('unconfirmed Stop preserves the last paired preview without counting it as a final', async () => {
  const h = harness();
  const socket = await h.start();
  await h.port(new ArrayBuffer(3200));
  await socket.message({type: 'interim', en: 'unfinished', ru: 'незавершённое'});
  await h.click();
  await h.port({type: 'flushed'});
  await socket.message({type: 'error', code: 'transcription_incomplete'});
  socket.close(1011);
  await settle();
  const warning = h.ids.history.children[0];
  assert.match(warning.children[0].textContent, /nepotvrzený/);
  assert.equal(warning.children[1].en.textContent, 'unfinished');
  assert.equal(warning.children[1].ru.textContent, 'незавершённое');
  assert.equal(h.ids['stat-final'].textContent, '0');
  assert.equal(h.ids['live-en'].textContent, '');
  await h.start();
  assert.equal(h.ids.history.children[0], warning);
});

test('a final draining after a completion warning replaces the preview, not duplicated as unconfirmed', async () => {
  const h = harness();
  const socket = await h.start();
  await h.port(new ArrayBuffer(3200));
  await socket.message({type: 'interim', en: 'draft', ru: 'черновик'});
  await h.click();
  await h.port({type: 'flushed'});
  await socket.message({type: 'error', code: 'transcription_incomplete'});
  const warning = h.ids.history.children[0];
  await socket.message({type: 'final', en: 'complete', ru: 'готово'});
  socket.close(1011);
  await settle();
  assert.equal(warning.children.length, 0);
  assert.equal(h.ids.history.children.length, 2);
  assert.equal(h.ids['stat-final'].textContent, '1');
});
