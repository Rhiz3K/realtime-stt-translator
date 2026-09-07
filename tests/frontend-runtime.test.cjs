// Run the actual presentation and audio modules against deterministic browser APIs.
// No microphone, server, credentials, or paid provider requests are used.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

const scripts = ['translator-ui.js', 'audio-session.js'].map(name => ({
  name, source: readFileSync(join(__dirname, '../app/static', name), 'utf8')
}));
const settle = () => new Promise(resolve => setImmediate(resolve));

function harness({saved = null, storageBlocked = false, fullscreenFails = false} = {}) {
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
    append(value) { this.children.push(value); }
    prepend(value) { this.children.unshift(value); this.onPrepend?.(value); }
    async dispatch(type, event = {}) {
      for (const listener of this.listeners[type] || []) await listener(event);
      await settle();
    }
  }

  const ids = Object.fromEntries([
    'toggle', 'status', 'live-panel', 'live-en', 'live-ru', 'history', 'result-template',
    'history-scroll', 'history-empty', 'history-count', 'live-note', 'font-size',
    'font-down', 'font-up', 'font-value', 'follow-latest', 'fullscreen-toggle',
    'compact-toggle', 'theme-toggle', 'toolbar', 'presentation-notice',
    'stat-audio', 'stat-interim', 'stat-final', 'stat-stt', 'stat-translation',
    'stat-queue', 'stat-hypotheses', 'stat-connect', 'audio-level'
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
  let nextTimer = 0;

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
    performance: {now: () => 1234},
    matchMedia: () => ({matches: false, addEventListener() {}}),
    location: {protocol: 'https:', host: 'translator.test'},
    navigator: {mediaDevices: {getUserMedia: async () => {
      const track = {stopped: false, stop() { this.stopped = true; }};
      tracks.push(track);
      return {getTracks: () => [track]};
    }}},
    WebSocket: Socket, AudioContext: Context, AudioWorkletNode: Worklet,
    ArrayBuffer, DataView, console,
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    addEventListener() {}
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const script of scripts) vm.runInContext(script.source, sandbox, {filename: script.name});

  return {
    ids, sockets, worklets, contexts, tracks, timers, root, document, storage,
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
  assert.match(h.ids.status.textContent, /Úplnost/);
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
