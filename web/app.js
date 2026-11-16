// UI for the in-browser music generator: generate, draw, play and download.
const { MusicLSTM, tokensToNotes } = window.MusicModel;
const $ = (id) => document.getElementById(id);

const HARMONICS = [1.0, 0.45, 0.25, 0.12, 0.06]; // same tone as synth.py
const SAMPLE_RATE = 22050;

let model = null;
let piece = null;       // { tokens, seed, notes, totalBeats }
let audio = null;       // { ctx, startTime, raf }

// ---------------------------------------------------------------- model
async function loadModel() {
  const [meta, buffer] = await Promise.all([
    fetch("model.json").then((r) => r.json()),
    fetch("model.bin").then((r) => r.arrayBuffer()),
  ]);
  model = new MusicLSTM(meta, buffer);
  const t = meta.training;
  $("model-stats").textContent =
    `Model: ${meta.vocab.length} tokens, best checkpoint from epoch ${t.epoch} ` +
    `(validation loss ${t.val_loss}, next-note accuracy ${(t.val_acc * 100).toFixed(0)}%).`;
  $("generate").disabled = false;
  $("generate").textContent = "🎲 Generate";
}

function generate() {
  stop();
  const btn = $("generate");
  btn.disabled = true;
  btn.textContent = "Composing…";
  // let the button repaint before the CPU-heavy part
  setTimeout(() => {
    const seeds = model.meta.seeds;
    const seed = seeds[Math.floor(Math.random() * seeds.length)];
    const t0 = performance.now();
    const tokens = model.generate(seed, Number($("length").value), {
      temperature: Number($("temperature").value),
      topK: 8,
    });
    const ms = performance.now() - t0;
    piece = { tokens, seed, ...tokensToNotes(tokens) };

    const pitches = piece.notes.flatMap((n) => n.pitches);
    $("info").textContent = `${piece.notes.length} notes · ${piece.totalBeats} beats · composed in ${(ms / 1000).toFixed(1)} s`;
    $("tokens").textContent = `Tokens: ${tokens.slice(0, 24).join("  ")}  …`;
    drawRoll(0, pitches);
    btn.disabled = false;
    btn.textContent = "🎲 Generate again";
    for (const id of ["play", "download-midi", "download-wav"]) $(id).disabled = false;
    play();
  }, 30);
}

// ---------------------------------------------------------------- piano roll
function drawRoll(playheadBeat = 0) {
  const canvas = $("roll");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const height = canvas.clientHeight || 260;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  g.clearRect(0, 0, width, height);
  if (!piece) return;

  const pitches = piece.notes.flatMap((n) => n.pitches);
  const lo = Math.min(...pitches) - 2;
  const hi = Math.max(...pitches) + 2;
  const rowH = height / (hi - lo + 1);
  const beatW = width / Math.max(piece.totalBeats, 1);

  g.strokeStyle = color("--grid");
  g.lineWidth = 1;
  for (let b = 0; b <= piece.totalBeats; b += 4) {
    const x = Math.round(b * beatW) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, height); g.stroke();
  }

  for (const n of piece.notes) {
    const played = n.start < playheadBeat;
    g.fillStyle = played ? color("--accent") : color("--note");
    for (const p of n.pitches) {
      const y = height - (p - lo + 1) * rowH;
      g.fillRect(n.start * beatW + 0.5, y + 1, Math.max(n.beats * beatW - 1.5, 1.5), Math.max(rowH - 2, 2));
    }
  }

  if (playheadBeat > 0) {
    g.fillStyle = color("--text");
    g.fillRect(playheadBeat * beatW, 0, 2, height);
  }
}

// ---------------------------------------------------------------- sound
function makeWave(ctx) {
  const real = new Float32Array(HARMONICS.length + 1);
  const imag = new Float32Array(HARMONICS.length + 1);
  HARMONICS.forEach((amp, i) => (imag[i + 1] = amp));
  return ctx.createPeriodicWave(real, imag);
}

function schedule(ctx, startAt, bpm) {
  const secPerBeat = 60 / bpm;
  const wave = makeWave(ctx);
  const master = ctx.createGain();
  master.gain.value = 0.5;
  const comp = ctx.createDynamicsCompressor();
  master.connect(comp).connect(ctx.destination);

  for (const n of piece.notes) {
    const start = startAt + n.start * secPerBeat;
    const dur = n.beats * secPerBeat;
    for (const midi of n.pitches) {
      const osc = ctx.createOscillator();
      osc.setPeriodicWave(wave);
      osc.frequency.value = 440 * 2 ** ((midi - 69) / 12);
      const env = ctx.createGain();
      const peak = 0.6 / Math.sqrt(n.pitches.length);
      env.gain.setValueAtTime(0, start);
      env.gain.linearRampToValueAtTime(peak, start + 0.01);   // attack
      env.gain.setTargetAtTime(0, start + 0.01, 0.4);         // piano-like decay
      env.gain.setTargetAtTime(0, start + dur, 0.08);         // release
      osc.connect(env).connect(master);
      osc.start(start);
      osc.stop(start + dur + 0.5);
    }
  }
  return secPerBeat;
}

function play() {
  if (!piece) return;
  stop();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const bpm = Number($("bpm").value);
  const startTime = ctx.currentTime + 0.1;
  const secPerBeat = schedule(ctx, startTime, bpm);
  audio = { ctx, startTime, secPerBeat, raf: 0 };
  $("play").innerHTML = "&#9632; Stop";

  const tick = () => {
    if (!audio) return;
    const beat = (audio.ctx.currentTime - audio.startTime) / audio.secPerBeat;
    if (beat > piece.totalBeats + 1) return stop();
    drawRoll(Math.max(beat, 0));
    audio.raf = requestAnimationFrame(tick);
  };
  tick();
}

function stop() {
  if (!audio) return;
  cancelAnimationFrame(audio.raf);
  audio.ctx.close();
  audio = null;
  $("play").innerHTML = "&#9654; Play";
  drawRoll(0);
}

// ---------------------------------------------------------------- downloads
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function toMidi(notes, bpm) {
  const TPQ = 480;
  const events = [];
  for (const n of notes) {
    for (const p of n.pitches) {
      events.push({ tick: Math.round(n.start * TPQ), on: true, p });
      events.push({ tick: Math.round((n.start + n.beats) * TPQ), on: false, p });
    }
  }
  events.sort((a, b) => a.tick - b.tick || a.on - b.on); // note-offs before note-ons

  const vlq = (value) => {
    const bytes = [value & 0x7f];
    while ((value >>= 7)) bytes.unshift((value & 0x7f) | 0x80);
    return bytes;
  };
  const tempo = Math.round(60000000 / bpm);
  const track = [0x00, 0xff, 0x51, 0x03, (tempo >> 16) & 255, (tempo >> 8) & 255, tempo & 255, 0x00, 0xc0, 0x00];
  let last = 0;
  for (const e of events) {
    track.push(...vlq(e.tick - last), e.on ? 0x90 : 0x80, e.p, e.on ? 80 : 0);
    last = e.tick;
  }
  track.push(0x00, 0xff, 0x2f, 0x00);

  const len = track.length;
  const header = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, (TPQ >> 8) & 255, TPQ & 255];
  const trackHeader = [0x4d, 0x54, 0x72, 0x6b, (len >>> 24) & 255, (len >>> 16) & 255, (len >>> 8) & 255, len & 255];
  return new Blob([new Uint8Array([...header, ...trackHeader, ...track])], { type: "audio/midi" });
}

function toWav(buffer) {
  const data = buffer.getChannelData(0);
  let peak = 0;
  for (const v of data) peak = Math.max(peak, Math.abs(v));
  const scale = peak > 0 ? 0.9 / peak : 1;
  const out = new DataView(new ArrayBuffer(44 + data.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + data.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, buffer.sampleRate, true); out.setUint32(28, buffer.sampleRate * 2, true);
  out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, data.length * 2, true);
  data.forEach((v, i) => out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v * scale)) * 32767, true));
  return new Blob([out], { type: "audio/wav" });
}

async function downloadWav() {
  const btn = $("download-wav");
  btn.disabled = true;
  btn.textContent = "Rendering…";
  const bpm = Number($("bpm").value);
  const seconds = piece.totalBeats * (60 / bpm) + 1.5;
  const ctx = new OfflineAudioContext(1, Math.ceil(seconds * SAMPLE_RATE), SAMPLE_RATE);
  schedule(ctx, 0.05, bpm);
  download(toWav(await ctx.startRendering()), "ai-music.wav");
  btn.disabled = false;
  btn.innerHTML = "&#11015; WAV";
}

// ---------------------------------------------------------------- wiring
for (const [id, out, fmt] of [["length", "length-value", (v) => v], ["temperature", "temp-value", (v) => Number(v).toFixed(1)], ["bpm", "bpm-value", (v) => v]]) {
  $(id).addEventListener("input", () => ($(out).textContent = fmt($(id).value)));
}
$("generate").addEventListener("click", generate);
$("play").addEventListener("click", () => (audio ? stop() : play()));
$("download-midi").addEventListener("click", () => download(toMidi(piece.notes, Number($("bpm").value)), "ai-music.mid"));
$("download-wav").addEventListener("click", downloadWav);
window.addEventListener("resize", () => drawRoll(0));

loadModel().catch((err) => {
  $("generate").textContent = "Could not load the model";
  $("info").textContent = err.message;
});
