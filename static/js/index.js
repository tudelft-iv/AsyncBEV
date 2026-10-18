'use strict';

// The manifest maps model names to verified assets; directory names are not model labels.
function validateSceneManifest(manifest, baseURL) {
  if (!manifest || manifest.version !== 1 || !Array.isArray(manifest.scenes) || !manifest.scenes.length) {
    throw new Error('The scene list is empty or has an unsupported format.');
  }
  const seen = new Set();
  return manifest.scenes.map(function (scene) {
    if (!scene || !/^scene-\d{4}$/.test(scene.id) || seen.has(scene.id)) {
      throw new Error('The scene list contains an invalid or duplicate scene.');
    }
    seen.add(scene.id);
    const result = { id: scene.id };
    ['cmt', 'asyncbev'].forEach(function (model) {
      if (typeof scene[model] !== 'string') throw new Error('A scene is missing a model video.');
      const url = new URL(scene[model], baseURL);
      const videoRoot = new URL('./static/video/', baseURL);
      if (url.origin !== videoRoot.origin || !url.pathname.startsWith(videoRoot.pathname) || !url.pathname.endsWith('.mp4') || url.search || url.hash) {
        throw new Error('A scene has an invalid video path.');
      }
      result[model] = url.href;
    });
    return result;
  });
}

function formatVideoTime(seconds) {
  const whole = Math.floor(Number.isFinite(seconds) && seconds > 0 ? seconds : 0);
  return Math.floor(whole / 60) + ':' + String(whole % 60).padStart(2, '0');
}

function setupSceneBrowser() {
  const browser = document.getElementById('scene-browser');
  if (!browser) return;
  const selector = document.getElementById('scene-selector');
  const count = document.getElementById('scene-count');
  const previous = document.getElementById('previous-scene');
  const next = document.getElementById('next-scene');
  const play = document.getElementById('play-pause');
  const restart = document.getElementById('restart-videos');
  const position = document.getElementById('playback-position');
  const time = document.getElementById('playback-time');
  const speed = document.getElementById('playback-speed');
  const status = document.getElementById('comparison-status');
  const retry = document.getElementById('retry-scenes');
  const models = ['cmt', 'asyncbev'];
  const videos = models.map(function (model) { return document.getElementById(model + '-scene-video'); });
  const videoStatuses = models.map(function (model) { return document.getElementById(model + '-video-status'); });
  const links = models.map(function (model) { return document.getElementById(model + '-video-link'); });
  let scenes = [];
  let selectedIndex = 0;
  let session = null;
  let manifestRequest = 0;

  function enablePlayback(enabled) {
    [play, restart, position, speed].forEach(function (control) { control.disabled = !enabled; });
  }

  function setStatus(message) { status.textContent = message; }

  function isCurrent(state) { return state === session && !state.abort.signal.aborted; }

  function stop(state) {
    if (!state) return;
    state.running = false;
    state.command += 1;
    window.cancelAnimationFrame(state.frame);
    state.frame = 0;
    videos.forEach(function (video, index) {
      video.pause();
      if (state.ready[index] && !state.failed) videoStatuses[index].textContent = 'Paused · ' + formatVideoTime(video.duration);
    });
    play.textContent = 'Play both';
  }

  function updateTimeline(state) {
    const current = Math.min(videos[state.leader].currentTime || 0, state.duration);
    position.value = String(current);
    position.setAttribute('aria-valuetext', formatVideoTime(current) + ' of ' + formatVideoTime(state.duration));
    time.textContent = formatVideoTime(current) + ' / ' + formatVideoTime(state.duration);
  }

  function setVideoTime(state, index, value) {
    if (!state.ready[index]) return;
    const target = Math.max(0, Math.min(value, state.duration));
    if (Math.abs(videos[index].currentTime - target) < 0.025) return;
    state.expectedSeek[index] = target;
    videos[index].currentTime = target;
  }

  function seekBoth(value) {
    const state = session;
    if (!state || !state.ready.every(Boolean) || state.failed) return;
    videos.forEach(function (_, index) { setVideoTime(state, index, value); });
    updateTimeline(state);
  }

  function tick(state) {
    if (!isCurrent(state) || !state.running) return;
    const leader = videos[state.leader];
    const followerIndex = 1 - state.leader;
    const follower = videos[followerIndex];
    if (leader.ended || leader.currentTime >= state.duration) {
      stop(state);
      updateTimeline(state);
      setStatus('Comparison finished. Replay or choose another scene.');
      return;
    }
    if (!leader.seeking && !follower.seeking && leader.readyState >= 3 && follower.readyState >= 3 && Math.abs(leader.currentTime - follower.currentTime) > 0.12) {
      setVideoTime(state, followerIndex, leader.currentTime);
    }
    updateTimeline(state);
    state.frame = window.requestAnimationFrame(function () { tick(state); });
  }

  async function playBoth(leaderIndex) {
    const state = session;
    if (!state || !state.ready.every(Boolean) || state.failed) {
      videos.forEach(function (video) { video.pause(); });
      return;
    }
    if (state.running) return;
    state.leader = typeof leaderIndex === 'number' ? leaderIndex : state.leader;
    if (videos[state.leader].ended || videos[state.leader].currentTime >= state.duration - 0.05) seekBoth(0);
    setVideoTime(state, 1 - state.leader, videos[state.leader].currentTime);
    state.running = true;
    const command = ++state.command;
    play.textContent = 'Pause both';
    setStatus('Starting synchronized playback…');
    state.frame = window.requestAnimationFrame(function () { tick(state); });
    const results = await Promise.allSettled(videos.map(function (video) { return video.play(); }));
    if (!isCurrent(state) || command !== state.command) return;
    if (results.some(function (result) { return result.status === 'rejected'; })) {
      stop(state);
      setStatus('Playback could not start. Press Play both to retry, or open a video directly.');
    } else {
      setStatus('Playing ' + scenes[selectedIndex].id + ' · playback linked by elapsed time.');
    }
  }

  function markReady(state, index) {
    if (!isCurrent(state) || videos[index].readyState < 1) return;
    const duration = videos[index].duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      markError(state, index);
      return;
    }
    state.ready[index] = true;
    videoStatuses[index].textContent = 'Ready · ' + formatVideoTime(duration);
    if (!state.ready.every(Boolean) || state.failed) return;
    state.duration = Math.min.apply(null, videos.map(function (video) { return video.duration; }));
    position.max = String(state.duration);
    enablePlayback(true);
    browser.setAttribute('aria-busy', 'false');
    updateTimeline(state);
    const difference = Math.abs(videos[0].duration - videos[1].duration);
    setStatus(scenes[selectedIndex].id + ' ready. Press Play both to compare.' + (difference > 0.25 ? ' Videos have different lengths; shared playback ends with the shorter video.' : ''));
  }

  function markError(state, index) {
    if (!isCurrent(state)) return;
    state.failed = true;
    stop(state);
    enablePlayback(false);
    browser.setAttribute('aria-busy', 'false');
    videoStatuses[index].textContent = 'Video could not be loaded';
    videoStatuses[index].classList.add('is-error');
    setStatus('This comparison could not be loaded. Retry, choose another scene, or open the video directly.');
    retry.hidden = false;
  }

  function selectScene(index) {
    if (!scenes.length || index < 0 || index >= scenes.length) return;
    if (session) {
      session.abort.abort();
      stop(session);
    }
    selectedIndex = index;
    const scene = scenes[index];
    const state = { abort: new AbortController(), ready: [false, false], expectedSeek: [null, null], failed: false, running: false, duration: 0, leader: 0, frame: 0, command: 0 };
    session = state;
    const options = { signal: state.abort.signal };
    selector.value = scene.id;
    count.textContent = (index + 1) + ' / ' + scenes.length + ' scenes';
    previous.disabled = index === 0;
    next.disabled = index === scenes.length - 1;
    browser.setAttribute('aria-busy', 'true');
    retry.hidden = true;
    speed.value = '1';
    position.value = '0';
    position.max = '0';
    position.setAttribute('aria-valuetext', '0:00');
    time.textContent = '0:00 / 0:00';
    enablePlayback(false);
    setStatus('Loading ' + scene.id + '…');

    videos.forEach(function (video, videoIndex) {
      videoStatuses[videoIndex].textContent = 'Loading video metadata…';
      videoStatuses[videoIndex].classList.remove('is-error');
      links[videoIndex].href = scene[models[videoIndex]];
      links[videoIndex].hidden = false;
      video.playbackRate = 1;
      video.src = scene[models[videoIndex]];
      video.addEventListener('loadedmetadata', function () { markReady(state, videoIndex); }, options);
      video.addEventListener('error', function () { markError(state, videoIndex); }, options);
      video.addEventListener('play', function () { if (isCurrent(state) && !state.running) playBoth(videoIndex); }, options);
      video.addEventListener('pause', function () {
        if (!isCurrent(state) || !state.running) return;
        stop(state);
        updateTimeline(state);
        setStatus(video.ended ? 'Comparison finished. Replay or choose another scene.' : 'Paused · ' + scene.id);
      }, options);
      video.addEventListener('ended', function () {
        if (!isCurrent(state)) return;
        stop(state);
        updateTimeline(state);
        setStatus('Comparison finished. Replay or choose another scene.');
      }, options);
      video.addEventListener('seeking', function () {
        if (!isCurrent(state) || !state.ready.every(Boolean) || state.failed) return;
        const expected = state.expectedSeek[videoIndex];
        state.expectedSeek[videoIndex] = null;
        if (expected !== null && Math.abs(video.currentTime - expected) < 0.15) return;
        state.leader = videoIndex;
        setVideoTime(state, 1 - videoIndex, video.currentTime);
        updateTimeline(state);
      }, options);
      video.addEventListener('ratechange', function () {
        if (!isCurrent(state)) return;
        const rate = video.playbackRate;
        if (!Array.from(speed.options).some(function (option) { return Number(option.value) === rate; })) {
          speed.add(new Option(rate + '×', String(rate)));
        }
        speed.value = String(rate);
        videos.forEach(function (target) { if (target.playbackRate !== rate) target.playbackRate = rate; });
      }, options);
      video.addEventListener('waiting', function () {
        if (!isCurrent(state) || state.failed || !state.ready[videoIndex]) return;
        videoStatuses[videoIndex].textContent = 'Buffering…';
      }, options);
      video.addEventListener('stalled', function () {
        if (!isCurrent(state) || state.failed) return;
        videoStatuses[videoIndex].textContent = 'Waiting for video data…';
        setStatus('Video data is loading slowly. You can retry or choose another scene.');
        retry.hidden = false;
      }, options);
      video.addEventListener('playing', function () {
        if (isCurrent(state) && !state.failed) videoStatuses[videoIndex].textContent = 'Playing · ' + formatVideoTime(video.duration);
      }, options);
      video.addEventListener('canplay', function () {
        if (isCurrent(state) && state.ready[videoIndex] && !state.failed) videoStatuses[videoIndex].textContent = (state.running ? 'Playing' : 'Ready') + ' · ' + formatVideoTime(video.duration);
      }, options);
      video.addEventListener('timeupdate', function () {
        if (isCurrent(state) && videoIndex === state.leader && !state.running) updateTimeline(state);
      }, options);
      video.load();
    });
  }

  async function loadScenes() {
    const request = ++manifestRequest;
    retry.hidden = true;
    browser.setAttribute('aria-busy', 'true');
    setStatus('Loading the scene list…');
    try {
      const response = await fetch('./static/video/scenes.json', { cache: 'no-cache' });
      if (!response.ok) throw new Error('The scene list could not be downloaded.');
      const manifest = await response.json();
      if (request !== manifestRequest) return;
      scenes = validateSceneManifest(manifest, document.baseURI);
      selector.replaceChildren();
      scenes.forEach(function (scene) { selector.add(new Option(scene.id, scene.id)); });
      selector.disabled = false;
      selectScene(0);
    } catch (error) {
      if (request !== manifestRequest) return;
      browser.setAttribute('aria-busy', 'false');
      setStatus('The scene list could not be loaded. Please try again.');
      videoStatuses.forEach(function (label) { label.textContent = 'Scene list unavailable'; });
      retry.hidden = false;
    }
  }

  selector.addEventListener('change', function () { selectScene(scenes.findIndex(function (scene) { return scene.id === selector.value; })); });
  previous.addEventListener('click', function () { selectScene(selectedIndex - 1); });
  next.addEventListener('click', function () { selectScene(selectedIndex + 1); });
  retry.addEventListener('click', function () { if (scenes.length) selectScene(selectedIndex); else loadScenes(); });
  play.addEventListener('click', function () {
    if (session && session.running) {
      stop(session);
      setStatus('Paused · ' + scenes[selectedIndex].id);
    } else playBoth();
  });
  restart.addEventListener('click', function () { seekBoth(0); });
  position.addEventListener('input', function () { seekBoth(Number(position.value)); });
  speed.addEventListener('change', function () {
    const rate = Number(speed.value);
    videos.forEach(function (video) { video.playbackRate = rate; });
  });
  window.addEventListener('pagehide', function () { stop(session); });
  loadScenes();
}

function setupCitationCopy() {
  const button = document.getElementById('copy-citation');
  if (!button) return;
  button.addEventListener('click', async function () {
    const text = document.getElementById('citation-text').textContent;
    const status = document.getElementById('copy-status');
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      status.textContent = 'BibTeX copied.';
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('citation-text'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Use your browser’s Copy command.';
    }
  });
}

function setupResearchDropdown() {
  const dropdown = document.getElementById('research-dropdown');
  if (!dropdown) return;
  const summary = dropdown.querySelector('summary');
  dropdown.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && dropdown.open) {
      dropdown.open = false;
      summary.focus();
      event.preventDefault();
      event.stopPropagation();
    }
  });
  function closeWhenOutside(event) {
    if (dropdown.open && !dropdown.contains(event.target)) dropdown.open = false;
  }
  document.addEventListener('pointerdown', closeWhenOutside);
  document.addEventListener('focusin', closeWhenOutside);
}

if (typeof document !== 'undefined') {
  setupResearchDropdown();
  setupSceneBrowser();
  setupCitationCopy();
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { validateSceneManifest: validateSceneManifest, formatVideoTime: formatVideoTime, setupSceneBrowser: setupSceneBrowser };
}
