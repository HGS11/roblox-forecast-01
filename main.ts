// Roblox Up-and-Coming top-10 forecaster (Node 22, standard library only).
// Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
// Model: per-game rank diffusion pooled over horizons, with an exit hazard and
// a reserve left unnamed for games never seen. Deterministic: no clock, no
// randomness; snapshot timestamps only. No network, no model calls.
//
// Evidence from the kit history (34 archive reads, 2026-10-30..2026-11-07):
// rank change per hour has SD 0.33 in ranks 1-10, 0.66 in 11-20, 1.08 in
// 21-40 and 1.41 below, so diffusion widens with rank; log player-count
// growth correlates ~0 with rank change, so player counts are not used as a
// predictor; survival to the next read is 99.4% (top 10) down to 93.3%
// (ranks 31+), so an exit hazard is applied, strongest low on the chart.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// Share of each place deliberately left unnamed, for games not in the
// history or too weak to name: the unnamed share is split over every game
// on a read's chart, so it never scores the -9.21 floor.
const RESERVE = 0.3;
// Horizons (hours) pooled with equal weight: the forecast is scored at
// every hour of a 72-hour window, so early persistence and late diffusion
// both matter.
const HORIZONS = [4, 8, 16, 24, 36, 48, 60, 72];
// Rank-diffusion SD per hour by starting band, measured on the history.
const BAND_SIGMA: Array<[number, number]> = [[10, 0.35], [20, 0.7], [40, 1.1], [1e9, 1.4]];
// Exit hazard per hour by band, from next-read survival rates.
const BAND_HAZARD: Array<[number, number]> = [[10, 0.001], [30, 0.002], [1e9, 0.011]];
const MAX_P = 0.65;
const MIN_P = 0.0015;

interface Row {
  rank: number;
  universe_id: number;
}

interface Snapshot {
  at: string;
  up_and_coming: Row[];
  details: Record<string, { created?: string | null } | null> | null;
}

interface Obs {
  t: number;
  rank: number;
}

interface Game {
  id: number;
  obs: Obs[];
  createdMs: number | null;
}

function bandValue(bands: Array<[number, number]>, rank: number): number {
  for (const [cap, v] of bands) if (rank <= cap) return v;
  return bands[bands.length - 1]![1];
}

// Standard normal CDF via the Abramowitz-Stegun erf approximation.
function normCdf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return 0.5 * (1 + s * (1 - poly * Math.exp(-z * z)));
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, "task.json"), "utf8")) as {
  places: number;
  output: string;
  history_until?: string;
};
const lines = readFileSync(join(INPUT_DIR, "history.ndjson"), "utf8").split("\n").filter((l) => l !== "");
const snaps: Snapshot[] = lines.map((l) => JSON.parse(l) as Snapshot);
const endMs = task.history_until ? Date.parse(task.history_until) : Date.parse(snaps[snaps.length - 1]!.at);
const HOUR = 3600e3;

const games = new Map<number, Game>();
for (const s of snaps) {
  const t = Date.parse(s.at);
  for (const r of s.up_and_coming ?? []) {
    let g = games.get(r.universe_id);
    if (!g) {
      g = { id: r.universe_id, obs: [], createdMs: null };
      games.set(r.universe_id, g);
    }
    g.obs.push({ t, rank: r.rank });
  }
  if (s.details) {
    for (const [key, d] of Object.entries(s.details)) {
      const g = games.get(Number(key));
      if (g && d && d.created && g.createdMs === null) g.createdMs = Date.parse(d.created);
    }
  }
}

interface Forecast {
  id: number;
  anchor: number;
  velocity: number;
  sigmaHr: number;
  hazard: number;
  absentHrs: number;
  placeP: number[];
}

const forecasts: Forecast[] = [];
for (const g of games.values()) {
  const obs = [...g.obs].sort((a, b) => a.t - b.t);
  const last = obs[obs.length - 1]!;
  const absentHrs = Math.max(0, (endMs - last.t) / HOUR);
  if (absentHrs > 96) continue; // long gone: covered by the unnamed share
  // Anchor: last rank, pulled slightly toward a 36h exponentially weighted
  // mean rank to smooth single-read noise.
  let wSum = 0;
  let rSum = 0;
  for (const o of obs) {
    const w = Math.exp(-(endMs - o.t) / (36 * HOUR));
    wSum += w;
    rSum += w * o.rank;
  }
  const anchor = 0.65 * last.rank + 0.35 * (wSum > 0 ? rSum / wSum : last.rank);
  // Velocity: rank change over the last ~48h, clamped; rank trajectories
  // are noisy, but backtests on three held-out splits showed the unshrunk
  // recent trend (clamped at 0.6 ranks/hour) beat shrunk versions, so no
  // shrinkage is applied beyond the clamp.
  let velocity = 0;
  const winStart = endMs - 48 * HOUR;
  const early = obs.find((o) => o.t >= winStart) ?? obs[0]!;
  const dtHrs = (last.t - early.t) / HOUR;
  if (dtHrs >= 6) velocity = clamp((last.rank - early.rank) / dtHrs, -0.6, 0.6);
  // Volatility: blend the game's own hourly-change SD with its band prior.
  let sqSum = 0;
  let nCh = 0;
  for (let i = 1; i < obs.length; i++) {
    const dt = (obs[i]!.t - obs[i - 1]!.t) / HOUR;
    if (dt > 0 && dt <= 12) {
      const d = (obs[i]!.rank - obs[i - 1]!.rank) / dt;
      sqSum += d * d;
      nCh++;
    }
  }
  const ownSigma = nCh >= 3 ? Math.sqrt(sqSum / nCh) : NaN;
  const prior = bandValue(BAND_SIGMA, last.rank);
  const sigmaHr = clamp(Number.isNaN(ownSigma) ? prior : 0.5 * ownSigma + 0.5 * prior, 0.25, 1.6);
  // Exit hazard, rising as a game nears the 28-day age-out (672h).
  let hazard = bandValue(BAND_HAZARD, last.rank);
  const ageHrs = g.createdMs !== null ? (endMs - g.createdMs) / HOUR : (endMs - obs[0]!.t) / HOUR;
  if (ageHrs > 480) {
    const over = (ageHrs - 480) / 192;
    hazard *= g.createdMs !== null ? 1 + 2.5 * over : 1 + 0.5 * over;
  }
  const placeP: number[] = [];
  for (let k = 1; k <= task.places; k++) {
    let acc = 0;
    for (const h of HORIZONS) {
      const mu = clamp(anchor + velocity * h + 0.1 * absentHrs, 0.5, 130);
      const sigma = Math.max(1.2, sigmaHr * Math.sqrt(h) + 0.8);
      const surv = Math.exp(-hazard * h) * Math.exp(-0.02 * absentHrs);
      acc += surv * (normCdf((k + 0.5 - mu) / sigma) - normCdf((k - 0.5 - mu) / sigma));
    }
    placeP.push(acc / HORIZONS.length);
  }
  forecasts.push({ id: g.id, anchor, velocity, sigmaHr, hazard, absentHrs, placeP });
}

// Per-place normalisation: scale down to 1 - RESERVE when the raw mass
// exceeds it; never scale up (honest shortfall stays unnamed).
interface ProbRow {
  universe_id: number;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (let k = 1; k <= task.places; k++) {
  const rows = forecasts
    .map((f) => ({ id: f.id, p: f.placeP[k - 1]! }))
    .filter((r) => r.p >= MIN_P)
    .sort((a, b) => b.p - a.p || a.id - b.id);
  const sum = rows.reduce((s, r) => s + r.p, 0);
  const target = 1 - RESERVE;
  const scale = sum > target ? target / sum : 1;
  for (const r of rows.slice(0, 100)) {
    const p = Math.min(MAX_P, r.p * scale);
    if (p >= MIN_P) probabilities.push({ universe_id: r.id, place: k, p });
  }
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

// top10: highest total probability of holding any top-10 place over the
// pooled horizons. (A short-horizon-only variant was tried and reverted:
// it scored worse on top-10 hits/agreement on the held-out splits.)
const scored = forecasts
  .map((f) => ({ id: f.id, mass: f.placeP.reduce((s, p) => s + p, 0), anchor: f.anchor }))
  .sort((a, b) => b.mass - a.mass || a.anchor - b.anchor || a.id - b.id);
const top10: number[] = scored.slice(0, task.places).map((s) => s.id);
if (top10.length < task.places) {
  const lastChart = [...(snaps[snaps.length - 1]?.up_and_coming ?? [])]
    .sort((a, b) => a.rank - b.rank)
    .map((r) => r.universe_id);
  for (const id of lastChart) {
    if (top10.length >= task.places) break;
    if (!top10.includes(id)) top10.push(id);
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
