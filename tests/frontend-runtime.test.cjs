// Run the actual inline application script against deterministic browser APIs.
// No microphone, server, credentials, or paid provider requests are used.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

const html = readFileSync(join(__dirname, '../app/templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const settle = () => new Promise(resolve => setImmediate(resolve));

function harness() {
  class Element {
    constructor() {
      this.dataset = {};
      this.children = [];
      this.textContent = '';
      this.hidden = false;
      this.disabled = false;
      this.listeners = {};
    }
    addEventListener(type, listener) {
      (this.listeners[type] ||= []).push(listener);
    }
    setAttribute(name, value) { this[name] = value; }
    append(value) { this.children.push(value); }
    async dispatch(type, event = {}) {
      for (const listener of this.listeners[type] || []) await listener(event);
      await settle();
    }
  }

  const ids = Object.fromEntries([
    'toggle', 'status', 'live-panel', 'live-en', 'live-ru', 'history', 'result-template'
  ].map(id => [id, new Element()]));
  ids['live-panel'].hidden = true;
  ids['result-template'].content = {
    cloneNode() {
      const en = new Element();
      const ru = new Element();
      return {en, ru, querySelector: selector => selector === '.result-en' ? en : ru};
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
  const sandbox = {
    document: {getElementById: id => ids[id], createElement: () => new Element()},
    location: {protocol: 'https:', host: 'translator.test'},
    navigator: {mediaDevices: {getUserMedia: async () => {
      const track = {stopped: false, stop() { this.stopped = true; }};
      tracks.push(track);
      return {getTracks: () => [track]};
    }}},
    WebSocket: Socket, AudioContext: Context, AudioWorkletNode: Worklet,
    ArrayBuffer, console,
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    addEventListener() {}
  };
  sandbox.window = sandbox;
  vm.runInNewContext(script, sandbox, {filename: 'index-inline.js'});

  return {
    ids, sockets, worklets, contexts, tracks, timers,
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
  assert.equal(h.ids.history.children[0], gap);
  assert.equal(h.ids.history.children[1].en.textContent, 'Done');
  assert.equal(h.ids.history.children[1].ru.textContent, 'Готово');
});

test('reset clears speculative text while preserving final history and gaps', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'final', en: 'saved', ru: 'сохранено'});
  await socket.message({type: 'interim', en: 'old', ru: 'старое'});
  socket.close();
  await settle();
  assert.equal(h.ids['live-panel'].hidden, true);
  assert.equal(h.ids['live-en'].textContent, '');
  assert.equal(h.ids['live-ru'].textContent, '');
  await h.start();
  assert.equal(h.ids['live-panel'].hidden, true);
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

test('terminal error releases audio immediately and final cannot restore live status', async () => {
  const h = harness();
  const socket = await h.start();
  await socket.message({type: 'error', code: 'transcription_incomplete'});
  await socket.message({type: 'final', en: 'last accepted', ru: 'принято'});
  assert.equal(h.tracks[0].stopped, true);
  assert.equal(h.worklets[0].port.closed, true);
  assert.match(h.ids.status.textContent, /Úplnost/);
  assert.equal(h.ids.history.children[0].dataset.code, 'transcription_incomplete');
});
