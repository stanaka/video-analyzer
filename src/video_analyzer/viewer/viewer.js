(() => {
  "use strict";
  const {result, playback} = JSON.parse(document.getElementById("analysis-data").textContent);
  const $ = id => document.getElementById(id);
  const settings = result.provenance.configuration || {};
  const duration = result.media.duration;
  const analysisRange = result.provenance.analysis_range || {start: 0, end: duration};
  const rangeStart = analysisRange.start;
  const rangeEnd = analysisRange.end;
  const rangeDuration = rangeEnd - rangeStart;
  const ja = result.report_language === "ja";
  const messages = {
    play: ["Play", "再生"], pause: ["Pause", "一時停止"], mute: ["Mute", "消音"],
    unmute: ["Unmute", "消音を解除"], seek: ["Playback position", "再生位置"],
    volume: ["Volume", "音量"], speed: ["Playback speed", "再生速度"],
    fullscreen: ["Fullscreen", "全画面"], audio: ["Audio", "音声"],
    locate: ["Locate original video", "元の動画を選択"],
    timeline: ["Speaker timeline", "話者タイムライン"],
    timelineHint: ["Hover to read speech · Click to play from any point", "カーソルを合わせて発話を表示 · クリックした位置から再生"],
    fit: ["Fit", "全体表示"], zoom: ["Zoom", "拡大"],
    transcript: ["Transcript", "文字起こし"], scenes: ["Scenes", "シーン"],
    metadata: ["Analysis details & warnings", "解析の詳細・注意事項"],
    footer: ["Local report · Speech, speaker assignments, and scene descriptions are model estimates.",
      "ローカルレポート · 文字起こし・話者区分・映像の説明はモデルによる推定です。"],
    original: ["Original", "元の音声"], processed: ["Processed", "処理済み"],
    peak: ["Peak normalization", "ピーク正規化"], dynamic: ["Dynamic normalization", "動的正規化"],
    denoise: ["Denoising", "ノイズ除去"], unknown: ["Unassigned", "話者不明"],
    noTranscript: ["No transcript at this position.", "この位置に文字起こしはありません。"],
    disabledTranscript: ["Transcription was disabled.", "文字起こしは無効でした。"],
    noSpeech: ["No speech detected.", "発話は検出されませんでした。"],
    noAudio: ["No audio track", "音声トラックなし"],
    disabledDiarization: ["Diarization was disabled; showing transcript timing.", "話者区分は無効でした。文字起こしの時刻を表示しています。"],
    noTurns: ["No speaker turns detected.", "話者の発話区間は検出されませんでした。"],
    noScenes: ["No scenes available.", "シーンがありません。"],
    noDescription: ["Scene descriptions were disabled.", "映像の説明は無効でした。"],
    videoError: ["Video unavailable or unsupported. Locate the original MP4 to reconnect it.", "動画が見つからないか、再生できない形式です。元のMP4を選択してください。"],
    audioError: ["Audio unavailable or unsupported. Restore the companion .media folder or export again.", "音声が見つからないか、再生できない形式です。.mediaフォルダーを戻すか再出力してください。"],
    playError: ["Playback could not start. Press Play to try again.", "再生を開始できませんでした。再生ボタンで再試行してください。"],
    wrongVideo: ["This video's dimensions or duration do not match the report. Choose the original MP4.", "動画の寸法または長さがレポートと一致しません。元のMP4を選択してください。"],
    fullscreenError: ["Fullscreen is unavailable in this browser.", "このブラウザーでは全画面表示を利用できません。"],
  };
  const tr = key => messages[key][ja ? 1 : 0];
  const clamp = (number, low, high) => Math.max(low, Math.min(high, number));
  const time = seconds => {
    const ms = Math.round(Math.max(0, seconds) * 1000);
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor(ms / 60000) % 60;
    return (hours ? String(hours).padStart(2, "0") + ":" : "") +
      String(minutes).padStart(2, "0") + ":" +
      String(Math.floor(ms / 1000) % 60).padStart(2, "0") + "." +
      String(ms % 1000).padStart(3, "0");
  };
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function joinText(units) {
    let text = "";
    for (const unit of units) {
      const piece = unit.text;
      if (!text) text = piece.trimStart();
      else if (/\s$/.test(text) || /^\s/.test(piece) ||
        /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/.test(text.slice(-1) + piece.slice(0, 1)) ||
        /^[,.!?;:、。！？)\]}」』]/.test(piece)) text += piece;
      else text += " " + piece;
    }
    return text.trim();
  }
  // Prefix maximum ends make lookup work for overlapping and point intervals.
  function intervalIndex(items) {
    const sorted = [...items].sort((a, b) => a.start - b.start || a.end - b.end);
    let maximum = -Infinity;
    const ends = sorted.map(item => (maximum = Math.max(maximum, item.end)));
    return (start, end) => {
      let low = 0, high = ends.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (ends[middle] < start) low = middle + 1;
        else high = middle;
      }
      const found = [];
      for (let index = low; index < sorted.length && sorted[index].start <= end; index++) {
        const item = sorted[index];
        const withinEnd = start === end ? item.start <= end : item.start < end;
        if (withinEnd && (item.end > start || (item.start === item.end && item.start >= start))) {
          found.push(item);
        }
      }
      return found;
    };
  }
  document.querySelectorAll("[data-i18n]").forEach(node => {
    node.textContent = tr(node.dataset.i18n);
  });
  document.querySelectorAll("[data-label]").forEach(node => {
    node.setAttribute("aria-label", tr(node.dataset.label));
  });
  $("player").setAttribute("aria-label", ja ? "動画プレーヤー" : "Video player");
  $("title").textContent = result.media.path.split(/[\\/]/).pop();
  $("summary").textContent = time(duration) + " · " + result.media.width + " × " +
    result.media.height + " · " + result.speakers.length + (ja ? " 話者" : " speakers");
  if (result.provenance.analysis_range) {
    $("summary").textContent += " · " + (ja ? "解析範囲 " : "Analysis range ") +
      time(rangeStart) + " – " + time(rangeEnd);
  }
  $("metadata").textContent = JSON.stringify({
    media: result.media, application_version: result.application_version,
    schema_version: result.schema_version, provenance: result.provenance
  }, null, 2);
  result.warnings.forEach(warning => $("warnings").append(element("li", "", warning)));
  $("warnings").hidden = !result.warnings.length;

  const video = $("video");
  const audioMode = $("audio-mode");
  const audioSources = new Map();
  audioMode.append(new Option(tr("original"), "original"));
  if (playback.processed) {
    const modes = [];
    if (playback.normalization !== "off") modes.push(tr(playback.normalization));
    if (playback.denoise) modes.push(tr("denoise"));
    audioMode.append(new Option(tr("processed") + " · " + modes.join(" + "), "processed"));
  }
  if (result.media.audio_stream_index === null) {
    audioMode.options[0].textContent = tr("noAudio");
    audioMode.disabled = $("volume").disabled = $("mute").disabled = true;
  }
  for (const name of ["original", "processed"]) {
    if (!playback[name]) continue;
    const audio = element("audio");
    audio.id = name + "-audio";
    audio.preload = "auto";
    audio.src = playback[name];
    audio.hidden = true;
    $("player").append(audio);
    audioSources.set(name, audio);
  }
  const selectedAudio = () => audioSources.get(audioMode.value) || null;
  const videoStart = Math.max(0, playback.video_start);
  const audioStart = playback.audio_start || 0;
  const currentTime = () => clamp(video.currentTime - videoStart, rangeStart, rangeEnd);
  let wantsPlayback = false;
  let generation = 0;
  let starting = null;
  let muted = false;
  let frame = null;
  let pendingTime = rangeStart;
  let objectUrl = null;
  let lastCorrection = 0;
  let timelineWidth = 1;
  let labelWidth = 132;
  let activeNodes = new Set();
  let lookupActive = () => [];
  const expectedPauses = new Map();

  function showError(message) {
    $("media-error").textContent = message;
    $("media-error").hidden = false;
  }
  function pauseMedia(media) {
    if (!media.paused) {
      expectedPauses.set(media, (expectedPauses.get(media) || 0) + 1);
      media.pause();
    }
  }
  function stopMedia() {
    pauseMedia(video);
    audioSources.forEach(pauseMedia);
  }
  function routeAudio() {
    const selected = selectedAudio();
    video.muted = muted || selected !== null || result.media.audio_stream_index === null;
    video.volume = Number($("volume").value);
    video.playbackRate = Number($("speed").value);
    audioSources.forEach(audio => {
      audio.muted = muted || audio !== selected;
      audio.volume = video.volume;
      audio.playbackRate = video.playbackRate;
      if (audio !== selected) pauseMedia(audio);
    });
    $("mute").textContent = tr(muted ? "unmute" : "mute");
    $("mute").setAttribute("aria-pressed", String(muted));
  }
  function updatePosition() {
    const position = currentTime();
    $("clock").textContent = time(position) + " / " + time(rangeEnd);
    $("seek").value = String(position);
    $("seek").setAttribute("aria-valuetext", time(position));
    $("play").textContent = tr(wantsPlayback ? "pause" : "play");
    $("playhead").style.left = (labelWidth + (position - rangeStart) / rangeDuration * timelineWidth) + "px";
    const next = new Set(lookupActive(position, position).map(item => item.node));
    activeNodes.forEach(node => { if (!next.has(node)) node.classList.remove("active"); });
    next.forEach(node => { if (!activeNodes.has(node)) node.classList.add("active"); });
    activeNodes = next;
  }
  function pause() {
    wantsPlayback = false;
    generation++;
    starting = null;
    stopMedia();
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    updatePosition();
  }
  function fail(message) {
    pause();
    showError(message);
  }
  function alignAudio(position = currentTime()) {
    const audio = selectedAudio();
    if (audio && audio.readyState >= 1) {
      audio.currentTime = clamp(position - audioStart, 0,
        Number.isFinite(audio.duration) ? audio.duration : rangeDuration);
    }
  }
  function tick() {
    frame = null;
    if (!wantsPlayback) return;
    if (currentTime() >= rangeEnd - 0.001) { seekTo(rangeEnd, false); return; }
    const audio = selectedAudio();
    if (audio && !audio.paused && !video.paused && !video.seeking && !audio.seeking &&
      Math.abs(audio.currentTime + audioStart - currentTime()) > 0.1 && performance.now() - lastCorrection > 250) {
      lastCorrection = performance.now();
      alignAudio();
    }
    updatePosition();
    frame = requestAnimationFrame(tick);
  }
  async function startPlayback(token = generation) {
    if (!wantsPlayback || token !== generation || starting === token) return;
    const audio = selectedAudio();
    if (video.error || (audio && audio.error)) {
      fail(tr(video.error ? "videoError" : "audioError"));
      return;
    }
    starting = token;
    routeAudio();
    if (frame === null) frame = requestAnimationFrame(tick);
    // Start both promises in the user gesture. Stale completions must not undo
    // a more recent seek, pause, or audio selection.
    try {
      await Promise.all([video.play(), ...(audio ? [audio.play()] : [])]);
    } catch (error) {
      if (token === generation && wantsPlayback) fail(tr("playError"));
    } finally {
      if (starting === token) starting = null;
    }
  }
  function seekTo(position, playAfter = wantsPlayback) {
    generation++;
    starting = null;
    stopMedia();
    wantsPlayback = playAfter;
    pendingTime = clamp(position, rangeStart, rangeEnd);
    if (video.readyState >= 1) video.currentTime = pendingTime + videoStart;
    alignAudio(pendingTime);
    $("media-error").hidden = true;
    updatePosition();
    if (playAfter) startPlayback();
  }
  function readyToResume() {
    const audio = selectedAudio();
    if (wantsPlayback && video.readyState >= 3 && !video.seeking &&
      (!audio || (audio.readyState >= 3 && !audio.seeking))) startPlayback();
  }
  function buffer(event) {
    if (!wantsPlayback || (event.target !== video && event.target !== selectedAudio())) return;
    generation++;
    starting = null;
    stopMedia();
    readyToResume();
  }
  function togglePlayback() {
    if (wantsPlayback) pause();
    else seekTo(currentTime() >= rangeEnd - 0.01 ? rangeStart : currentTime(), true);
  }
  $("seek").min = String(rangeStart);
  $("seek").max = String(rangeEnd);
  $("seek").addEventListener("input", () => seekTo(Number($("seek").value)));
  $("play").addEventListener("click", togglePlayback);
  video.addEventListener("click", togglePlayback);
  $("mute").addEventListener("click", () => { muted = !muted; routeAudio(); });
  $("volume").addEventListener("input", routeAudio);
  $("speed").addEventListener("change", routeAudio);
  audioMode.addEventListener("change", () => {
    const position = currentTime();
    generation++;
    starting = null;
    stopMedia();
    routeAudio();
    alignAudio(position);
    $("media-error").hidden = true;
    if (selectedAudio()?.error) fail(tr("audioError"));
    else if (wantsPlayback) startPlayback();
  });
  video.addEventListener("loadedmetadata", () => {
    expectedPauses.delete(video);
    if (video.videoWidth !== result.media.width || video.videoHeight !== result.media.height ||
      video.duration < videoStart + duration - 0.1) {
      fail(tr("wrongVideo")); return;
    }
    video.currentTime = pendingTime + videoStart;
    alignAudio(pendingTime);
    updatePosition();
  });
  video.addEventListener("error", () => fail(tr("videoError")));
  video.addEventListener("ended", pause);
  video.addEventListener("timeupdate", () => {
    const position = video.currentTime - videoStart;
    if (video.readyState >= 1 && (position < rangeStart || position > rangeEnd)) {
      seekTo(clamp(position, rangeStart, rangeEnd), position < rangeStart && wantsPlayback);
    } else updatePosition();
  });
  for (const media of [video, ...audioSources.values()]) {
    media.addEventListener("pause", () => {
      const expected = expectedPauses.get(media) || 0;
      if (expected) expectedPauses.set(media, expected - 1);
      else if (media.paused && (media === video || media === selectedAudio())) pause();
    });
    media.addEventListener("play", () => {
      // Respect playback actions from browser/OS media controls as well.
      if (!wantsPlayback && !media.paused && (media === video || media === selectedAudio())) {
        wantsPlayback = true;
        generation++;
        alignAudio();
        startPlayback();
      }
    });
    media.addEventListener("waiting", buffer);
    media.addEventListener("canplay", readyToResume);
    media.addEventListener("seeked", readyToResume);
  }
  audioSources.forEach(audio => {
    audio.addEventListener("ended", () => {
      if (audio === selectedAudio() && wantsPlayback) pause();
    });
    audio.addEventListener("loadedmetadata", () => {
      if (audio === selectedAudio()) alignAudio(currentTime());
    });
    audio.addEventListener("error", () => {
      if (audio === selectedAudio()) fail(tr("audioError"));
    });
  });
  $("fullscreen").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $("player").requestFullscreen();
    } catch { showError(tr("fullscreenError")); }
  });
  $("video-file").addEventListener("change", () => {
    const file = $("video-file").files[0];
    if (!file) return;
    pause();
    pendingTime = currentTime();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    $("media-error").hidden = true;
    video.src = objectUrl;
    video.load();
  });
  window.addEventListener("pagehide", () => {
    pause();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  });

  const palette = ["#74c9b4", "#93b4ef", "#d2a0df", "#e2be7c", "#df9296", "#81c4d9", "#b8c982", "#b5a7ed"];
  const speakers = settings.diarization === false ? [] :
    [...new Set([...result.speakers.map(speaker => speaker.id), ...result.diarization.map(turn => turn.speaker_id)])];
  const colors = new Map(speakers.map((speaker, index) => [speaker, palette[index % palette.length]]));
  const speakerName = id => id || tr("unknown");
  const units = settings.transcription === false ? [] : result.transcript;
  const unitsBySpeaker = new Map();
  for (const unit of units) {
    const key = settings.diarization === false ? null : unit.speaker_id;
    if (!unitsBySpeaker.has(key)) unitsBySpeaker.set(key, []);
    unitsBySpeaker.get(key).push(unit);
  }
  const speechLookup = new Map([...unitsBySpeaker].map(([key, values]) => [key, intervalIndex(values)]));
  const blockData = new WeakMap();
  const trackSpeakers = new WeakMap();
  const activeItems = [];
  const transcriptEntries = [];
  let lookupTranscript = () => [];
  function addTrack(name, color, scene = false) {
    const track = element("div", "track" + (scene ? " scene-track" : ""));
    track.style.setProperty("--speaker-color", color || "#8496a5");
    const label = element("div", "track-label");
    label.append(element("span", "swatch"), element("span", "", name));
    label.title = name;
    const body = element("div", "track-body");
    track.append(label, body);
    $("tracks").append(track);
    return body;
  }
  function addBlock(body, span, scene = false, text = "") {
    const start = clamp(span.start, rangeStart, rangeEnd), end = clamp(span.end, start, rangeEnd);
    const block = element("button", scene ? "scene-marker" : "turn", text);
    block.type = "button";
    block.style.left = ((start - rangeStart) / rangeDuration * 100) + "%";
    block.style.width = ((end - start) / rangeDuration * 100) + "%";
    block.setAttribute("aria-label", (scene ? tr("scenes") : speakerName(span.speaker_id)) +
      " " + time(start) + " – " + time(end));
    blockData.set(block, span);
    block.addEventListener("focus", () => {
      if (scene) return;
      const rect = block.getBoundingClientRect();
      showTooltip(span, start, rect.left, rect.bottom);
    });
    block.addEventListener("blur", hideTooltip);
    body.append(block);
    if (!scene) activeItems.push({...span, node: block});
  }
  if (result.scenes.length) {
    const body = addTrack(tr("scenes"), null, true);
    result.scenes.forEach((scene, index) => addBlock(body, scene, true, String(index + 1)));
  }
  for (const speaker of speakers) {
    const body = addTrack(speaker, colors.get(speaker));
    trackSpeakers.set(body, speaker);
    result.diarization.filter(turn => turn.speaker_id === speaker).forEach(turn => addBlock(body, turn));
  }
  const neutralUnits = settings.diarization === false ? units : units.filter(unit => !unit.speaker_id);
  if (neutralUnits.length) {
    const body = addTrack(settings.diarization === false ? tr("transcript") : tr("unknown"));
    trackSpeakers.set(body, null);
    neutralUnits.forEach(unit => addBlock(body, {...unit, speaker_id: null}));
  }
  if (settings.diarization === false || !result.diarization.length) {
    $("timeline-empty").hidden = false;
    $("timeline-empty").textContent = tr(settings.diarization === false ? "disabledDiarization" : "noTurns");
  }
  function positionAt(event) {
    const body = event.target.closest(".track-body, .ruler");
    if (!body) return null;
    const rect = body.getBoundingClientRect();
    return clamp(rangeStart + (event.clientX - rect.left) / rect.width * rangeDuration,
      rangeStart, rangeEnd);
  }
  function hideTooltip() { $("tooltip").hidden = true; }
  function showTooltip(span, position, x, y) {
    const start = Math.max(span.start, position - 5);
    const end = Math.min(span.end, position + 5);
    const nearby = speechLookup.get(span.speaker_id)?.(start, end) || [];
    $("tooltip-title").textContent = speakerName(span.speaker_id) + " · " + time(position) +
      " · " + time(span.start) + " – " + time(span.end);
    $("tooltip-text").textContent = joinText(nearby) || tr(settings.transcription === false ? "disabledTranscript" : "noTranscript");
    const tooltip = $("tooltip");
    tooltip.hidden = false;
    const rect = tooltip.getBoundingClientRect();
    tooltip.style.left = clamp(x + 14, 8, Math.max(8, innerWidth - rect.width - 8)) + "px";
    tooltip.style.top = clamp(y + 16, 8, Math.max(8, innerHeight - rect.height - 8)) + "px";
  }
  $("timeline").addEventListener("pointermove", event => {
    const block = event.target.closest(".turn");
    if (!block) { hideTooltip(); return; }
    showTooltip(blockData.get(block), positionAt(event), event.clientX, event.clientY);
  });
  $("timeline").addEventListener("pointerleave", hideTooltip);
  $("timeline-scroll").addEventListener("scroll", hideTooltip);
  function scrollTranscript(position, speaker) {
    const matches = lookupTranscript(position, position);
    const entry = (speaker !== undefined && matches.find(item => item.speaker_id === speaker)) ||
      matches[0] || transcriptEntries.find(item => item.start >= position) || transcriptEntries.at(-1);
    if (!entry) return;
    const pane = $("transcript");
    const row = entry.node.getBoundingClientRect();
    // Scroll only the transcript pane, keeping the player and timeline in place.
    pane.scrollTop += row.top - pane.getBoundingClientRect().top - pane.clientTop -
      (pane.clientHeight - row.height) / 2;
  }
  $("timeline").addEventListener("click", event => {
    const block = event.target.closest(".turn, .scene-marker");
    const position = event.detail === 0 && block ? blockData.get(block).start : positionAt(event);
    if (position !== null) {
      seekTo(position, true);
      scrollTranscript(position, trackSpeakers.get(event.target.closest(".track-body")));
    }
  });
  $("timeline-scroll").addEventListener("keydown", event => {
    if (event.target !== $("timeline-scroll")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      seekTo(currentTime() + (event.key === "ArrowRight" ? 5 : -5));
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });
  function layoutTimeline() {
    labelWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--label-width"));
    const viewport = Math.max(1, $("timeline-scroll").clientWidth - labelWidth);
    timelineWidth = viewport * Number($("zoom").value);
    $("timeline").style.width = (timelineWidth + labelWidth) + "px";
    $("ruler").replaceChildren();
    const rawStep = rangeDuration / Math.max(1, timelineWidth / 100);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 5, 10].map(value => value * magnitude).find(value => value >= rawStep);
    const ticks = document.createDocumentFragment();
    for (let index = 0; index * step < rangeDuration; index++) {
      const position = rangeStart + index * step;
      const tick = element("span", "tick", time(position).replace(/\.000$/, ""));
      tick.style.left = ((position - rangeStart) / rangeDuration * 100) + "%";
      ticks.append(tick);
    }
    $("ruler").append(ticks);
    updatePosition();
  }
  $("zoom").addEventListener("input", () => {
    const scroll = $("timeline-scroll");
    const center = (scroll.scrollLeft + (scroll.clientWidth - labelWidth) / 2) / timelineWidth;
    layoutTimeline();
    scroll.scrollLeft = center * timelineWidth - (scroll.clientWidth - labelWidth) / 2;
  });
  $("fit").addEventListener("click", () => {
    $("zoom").value = "1";
    layoutTimeline();
    $("timeline-scroll").scrollLeft = 0;
  });
  new ResizeObserver(layoutTimeline).observe($("timeline-scroll"));

  const groups = [];
  for (const unit of [...units].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const previous = groups[groups.length - 1];
    if (previous && previous.speaker_id === unit.speaker_id && unit.start - previous.end <= 1.5 &&
      unit.end - previous.start <= 20) {
      previous.end = Math.max(previous.end, unit.end);
      previous.units.push(unit);
    } else groups.push({...unit, units: [unit]});
  }
  if (!groups.length) $("transcript").append(element("p", "empty",
    tr(settings.transcription === false ? "disabledTranscript" : "noSpeech")));
  for (const group of groups) {
    const row = element("button", "transcript-entry");
    row.type = "button";
    const meta = element("span", "entry-meta");
    const swatch = element("span", "swatch");
    swatch.style.setProperty("--speaker-color", colors.get(group.speaker_id) || "#8496a5");
    meta.append(swatch, element("span", "", time(group.start) + " – " + time(group.end)));
    if (settings.diarization !== false) meta.append(element("span", "", speakerName(group.speaker_id)));
    row.append(meta, element("span", "entry-text", joinText(group.units)));
    row.addEventListener("click", () => seekTo(group.start, true));
    $("transcript").append(row);
    const entry = {...group, node: row};
    transcriptEntries.push(entry);
    activeItems.push(entry);
  }
  result.scenes.forEach((scene, index) => {
    const row = element("div", "scene-entry");
    const button = element("button", "", tr("scenes") + " " + (index + 1) + " · " +
      time(scene.start) + " – " + time(scene.end));
    button.type = "button";
    button.addEventListener("click", () => seekTo(scene.start, true));
    row.append(button);
    if (settings.scene_description !== false) {
      if (scene.description != null) row.append(element("p", "", scene.description));
      else scene.observations.forEach(observation => {
        const item = element("p");
        const jump = element("button", "", time(observation.start));
        jump.type = "button";
        jump.addEventListener("click", () => seekTo(observation.start, true));
        item.append(jump, document.createTextNode(" · " + observation.description));
        row.append(item);
      });
    }
    $("scenes").append(row);
  });
  if (settings.scene_description === false) $("scenes").prepend(element("p", "empty", tr("noDescription")));
  if (!result.scenes.length) $("scenes").append(element("p", "empty", tr("noScenes")));
  lookupActive = intervalIndex(activeItems);
  lookupTranscript = intervalIndex(transcriptEntries);
  routeAudio();
  layoutTimeline();
  video.src = playback.video;
})();
