'use strict';

// No browser, network, media decoding, or GPU required. These checks cover state
// transitions; verify actual rendering and media playback in a browser as well.
const assert = require('node:assert/strict');
const { setMaxListeners } = require('node:events');
const { validateSceneManifest, formatVideoTime, setupSceneBrowser } = require(process.argv[2] || '../static/js/index.js');
setMaxListeners(0);

const manifest = {
  version: 1,
  scenes: ['scene-0003', 'scene-0016', 'scene-0039'].map(id => ({
    id,
    cmt: './static/video/cmt_async_10/' + id + '/LIDAR_TOP_with_GT_vfr.mp4',
    asyncbev: './static/video/asyncbev_async_10/' + id + '/LIDAR_TOP_with_GT_vfr.mp4'
  }))
};
const base = 'https://example.org/AsyncBEV/';
assert.equal(validateSceneManifest(manifest, base).length, 3);
assert.equal(validateSceneManifest(manifest, base)[0].cmt, base + 'static/video/cmt_async_10/scene-0003/LIDAR_TOP_with_GT_vfr.mp4');
assert.throws(() => validateSceneManifest({version: 1, scenes: []}, base));
assert.throws(() => validateSceneManifest({version: 2, scenes: manifest.scenes}, base));
assert.throws(() => validateSceneManifest({version: 1, scenes: [manifest.scenes[0], manifest.scenes[0]]}, base));
for (const path of ['https://other.example/movie.mp4', './static/video/../../private.mp4', './static/video/movie.txt', './static/video/movie.mp4?x=1']) {
  assert.throws(() => validateSceneManifest({version: 1, scenes: [{...manifest.scenes[0], cmt: path}]}, base));
}
assert.equal(formatVideoTime(61.9), '1:01');
assert.equal(formatVideoTime(NaN), '0:00');
assert.equal(formatVideoTime(-2), '0:00');

class Element extends EventTarget {
  constructor(id) {
    super(); this.id = id; this.disabled = false; this.hidden = false; this.value = ''; this.textContent = ''; this.options = []; this.attributes = new Map();
    const classes = new Set();
    this.classList = {add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name)};
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.get(name); }
  add(option) { this.options.push(option); }
  replaceChildren() { this.options = []; }
  trigger(type) { this.dispatchEvent(new Event(type)); }
}
class Video extends Element {
  constructor(id) { super(id); this._currentTime = 0; this._rate = 1; this.paused = true; this.readyState = 0; this.duration = NaN; this.ended = false; this.seeking = false; this.loads = 0; }
  get currentTime() { return this._currentTime; }
  set currentTime(value) { this._currentTime = value; this.trigger('seeking'); }
  get playbackRate() { return this._rate; }
  set playbackRate(value) { if (value === this._rate) return; this._rate = value; this.trigger('ratechange'); }
  load() { this.loads++; this._currentTime = 0; this.readyState = 0; this.duration = NaN; this.ended = false; }
  play() { if (this.paused) { this.paused = false; this.trigger('play'); } return Promise.resolve(); }
  pause() { if (!this.paused) { this.paused = true; this.trigger('pause'); } }
  metadata(duration = 20) { this.duration = duration; this.readyState = 4; this.trigger('loadedmetadata'); }
}
const ids = ['scene-browser','scene-selector','scene-count','previous-scene','next-scene','play-pause','restart-videos','playback-position','playback-time','playback-speed','comparison-status','retry-scenes','cmt-video-status','asyncbev-video-status','cmt-video-link','asyncbev-video-link'];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
const videos = ['cmt', 'asyncbev'].map(model => {
  const id = model + '-scene-video';
  return elements[id] = new Video(id);
});
elements['playback-speed'].options = [{value: '0.5'}, {value: '1'}, {value: '1.5'}, {value: '2'}];
global.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
global.document = {baseURI: base, getElementById: id => elements[id]};
let frameID = 0;
const frames = new Map();
global.window = new EventTarget();
window.requestAnimationFrame = fn => { const id = ++frameID; frames.set(id, fn); return id; };
window.cancelAnimationFrame = id => frames.delete(id);
let fetchCount = 0;
global.fetch = async () => { fetchCount++; return {ok: true, json: async () => manifest}; };
const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };

(async () => {
  setupSceneBrowser();
  await flush();
  assert.equal(fetchCount, 1);
  assert.equal(elements['scene-selector'].options.length, 3);
  assert.equal(elements['scene-count'].textContent, '1 / 3 scenes');
  assert.equal(elements['previous-scene'].disabled, true);
  assert.equal(videos[0].loads, 1);
  assert.match(videos[0].src, /scene-0003/);
  assert.equal(elements['play-pause'].disabled, true);
  assert.equal(videos[0].paused, true);
  videos[0].metadata();
  assert.equal(elements['play-pause'].disabled, true);
  videos[1].metadata();
  assert.equal(elements['play-pause'].disabled, false);
  assert.equal(elements['scene-browser'].getAttribute('aria-busy'), 'false');
  elements['play-pause'].trigger('click');
  await flush();
  assert.deepEqual(videos.map(video => video.paused), [false, false]);
  assert.equal(elements['play-pause'].textContent, 'Pause both');
  videos[1].playbackRate = 1.5;
  assert.equal(videos[0].playbackRate, 1.5);
  assert.equal(elements['playback-speed'].value, '1.5');
  videos[1].currentTime = 4;
  assert.equal(videos[0].currentTime, 4);
  assert.equal(elements['playback-position'].value, '4');
  videos[0].pause();
  assert.deepEqual(videos.map(video => video.paused), [true, true]);
  assert.match(elements['cmt-video-status'].textContent, /Paused/);
  videos[1].play();
  await flush();
  assert.deepEqual(videos.map(video => video.paused), [false, false]);
  elements['next-scene'].trigger('click');
  assert.equal(elements['scene-count'].textContent, '2 / 3 scenes');
  assert.deepEqual(videos.map(video => video.paused), [true, true]);
  assert.deepEqual(videos.map(video => video.currentTime), [0, 0]);
  assert.deepEqual(videos.map(video => video.playbackRate), [1, 1]);
  assert.equal(elements['play-pause'].disabled, true);
  videos[0].metadata(20);
  videos[1].metadata(19);
  assert.match(elements['comparison-status'].textContent, /different lengths/);
  assert.equal(elements['playback-position'].max, '19');
  elements['playback-position'].value = '7';
  elements['playback-position'].trigger('input');
  assert.deepEqual(videos.map(video => video.currentTime), [7, 7]);
  elements['restart-videos'].trigger('click');
  assert.deepEqual(videos.map(video => video.currentTime), [0, 0]);
  videos[0].trigger('error');
  assert.equal(elements['play-pause'].disabled, true);
  assert.equal(elements['retry-scenes'].hidden, false);
  assert.equal(elements['next-scene'].disabled, false);
  elements['retry-scenes'].trigger('click');
  videos.forEach(video => video.metadata());
  assert.equal(elements['play-pause'].disabled, false);
  elements['next-scene'].trigger('click');
  assert.equal(elements['scene-count'].textContent, '3 / 3 scenes');
  assert.equal(elements['next-scene'].disabled, true);
  assert.equal(elements['previous-scene'].disabled, false);
  videos.forEach(video => video.metadata());
  videos[0].play = () => Promise.reject(new Error('Playback blocked'));
  elements['play-pause'].trigger('click');
  await flush();
  assert.equal(elements['play-pause'].textContent, 'Play both');
  assert.deepEqual(videos.map(video => video.paused), [true, true]);
  assert.match(elements['comparison-status'].textContent, /could not start/);
  assert.equal(frames.size, 0);
  console.log('Scene manifest, shared and native playback, seeking, speed, reset, bounds, and error recovery checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
