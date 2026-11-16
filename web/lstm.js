/*
 * The music LSTM in plain JavaScript — the same network as model.py (PyTorch):
 *   token -> Embedding -> LSTM layer 1 -> LSTM layer 2 -> Linear -> next-token logits
 * Weights come from model.bin, exported by export_web.py.
 */
(function (global) {
  "use strict";

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  class MusicLSTM {
    constructor(meta, buffer) {
      this.meta = meta;
      this.vocab = meta.vocab;
      this.cfg = meta.config;
      this.endId = this.vocab.indexOf("END");
      const all = new Float32Array(buffer);
      this.w = {};
      for (const t of meta.tensors) {
        const size = t.shape.reduce((a, b) => a * b, 1);
        this.w[t.name] = all.subarray(t.offset, t.offset + size);
      }
    }

    initialState() {
      const H = this.cfg.hidden_size;
      return Array.from({ length: this.cfg.num_layers }, () => ({ h: new Float32Array(H), c: new Float32Array(H) }));
    }

    // y = W x + b  (W is row-major [rows, cols])
    static affine(W, x, b, rows, out) {
      const cols = x.length;
      for (let r = 0; r < rows; r++) {
        let sum = b ? b[r] : 0;
        const base = r * cols;
        for (let c = 0; c < cols; c++) sum += W[base + c] * x[c];
        out[r] = sum;
      }
      return out;
    }

    // One time step. Returns the logits for the next token; `state` is updated in place.
    step(tokenId, state) {
      const { embed_dim: E, hidden_size: H, num_layers: L, vocab_size: V } = this.cfg;
      let x = this.w["embed.weight"].subarray(tokenId * E, (tokenId + 1) * E);
      const gi = new Float32Array(4 * H);
      const gh = new Float32Array(4 * H);

      for (let l = 0; l < L; l++) {
        const s = state[l];
        MusicLSTM.affine(this.w[`lstm.weight_ih_l${l}`], x, this.w[`lstm.bias_ih_l${l}`], 4 * H, gi);
        MusicLSTM.affine(this.w[`lstm.weight_hh_l${l}`], s.h, this.w[`lstm.bias_hh_l${l}`], 4 * H, gh);
        const h = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          // PyTorch gate order: input, forget, cell (g), output
          const i = sigmoid(gi[j] + gh[j]);
          const f = sigmoid(gi[H + j] + gh[H + j]);
          const g = Math.tanh(gi[2 * H + j] + gh[2 * H + j]);
          const o = sigmoid(gi[3 * H + j] + gh[3 * H + j]);
          s.c[j] = f * s.c[j] + i * g;
          h[j] = o * Math.tanh(s.c[j]);
        }
        s.h = h;
        x = h;
      }
      return MusicLSTM.affine(this.w["head.weight"], x, this.w["head.bias"], V, new Float32Array(V));
    }

    // Sample `length` new tokens after reading the seed phrase (temperature + top-k sampling).
    generate(seedTokens, length, { temperature = 1.0, topK = 8, random = Math.random } = {}) {
      const toId = new Map(this.vocab.map((t, i) => [t, i]));
      const state = this.initialState();
      let logits = null;
      for (const tok of seedTokens) logits = this.step(toId.get(tok), state);

      const out = [];
      for (let n = 0; n < length; n++) {
        const scaled = Array.from(logits, (v) => v / Math.max(temperature, 1e-3));
        scaled[this.endId] = -Infinity; // keep going, don't end the piece
        const order = scaled.map((v, i) => i).sort((a, b) => scaled[b] - scaled[a]).slice(0, topK);
        const maxLogit = scaled[order[0]];
        const weights = order.map((i) => Math.exp(scaled[i] - maxLogit));
        const total = weights.reduce((a, b) => a + b, 0);
        let pick = random() * total;
        let chosen = order[order.length - 1];
        for (let k = 0; k < order.length; k++) {
          pick -= weights[k];
          if (pick <= 0) { chosen = order[k]; break; }
        }
        out.push(this.vocab[chosen]);
        logits = this.step(chosen, state);
      }
      return out;
    }
  }

  // ---------------------------------------------------------------- token helpers (music_utils.py)
  const PITCH_CLASS = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  function pitchToMidi(name) {
    // music21 spelling: "C#4", "E-5" (flat), "B--3" (double flat)
    const m = /^([A-G])([#-]*)(-?\d+)$/.exec(name);
    if (!m) throw new Error(`Bad pitch ${name}`);
    const accidental = [...m[2]].reduce((a, ch) => a + (ch === "#" ? 1 : -1), 0);
    return (Number(m[3]) + 1) * 12 + PITCH_CLASS[m[1]] + accidental;
  }

  function parseToken(token) {
    const cut = token.lastIndexOf("_");
    const body = token.slice(0, cut);
    const beats = Number(token.slice(cut + 1));
    return { pitches: body === "R" ? [] : body.split(".").map(pitchToMidi), beats };
  }

  // [{start, beats, pitches}] in beats
  function tokensToNotes(tokens) {
    const notes = [];
    let t = 0;
    for (const tok of tokens) {
      if (tok === "END") continue;
      const { pitches, beats } = parseToken(tok);
      if (pitches.length) notes.push({ start: t, beats, pitches });
      t += beats;
    }
    return { notes, totalBeats: t };
  }

  const api = { MusicLSTM, parseToken, pitchToMidi, tokensToNotes };
  global.MusicModel = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
