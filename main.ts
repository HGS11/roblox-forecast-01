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
// Lifecycle drift: top games systematically fall (mean +6.5 ranks over
// 24-48h for ranks 1-10) while bottom games rise (-4.8 for ranks 41+).
// Centering the forecast mean on that drift map was tried and REJECTED:
// backtests showed it loses 1.5-3.5 points versus extrapolating each
// game's own recent trend, because the rank distribution is right-skewed
// and the mean is pulled up by collapsers while the typical game stays
// near its rank. The drift map is therefore used only to calibrate the
// residual-spread table (what remains after removing systematic drift),
// and the mean keeps the game's own velocity extrapolation. The spread
// table is scaled by 0.5: the full calibrated SDs are too wide for the
// log score on held-out splits, and 0.5 was the best of 0.45-0.8 swept.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// Share of each place deliberately left unnamed, for games not in the
// history or too weak to name: the unnamed share is split over every game
// on a read's chart, so it never scores the -9.21 floor. 0.25 was the best
// of 0.1-0.3 swept on held-out splits.
const RESERVE = 0.25;
// Horizons (hours) pooled with equal weight: the forecast is scored at
// every hour of a 72-hour window, so early persistence and late diffusion
// both matter.
const HORIZONS = [4, 8, 16, 24, 36, 48, 60, 72];
// Exit hazard per hour by band, from next-read survival rates.
const BAND_HAZARD: Array<[number, number]> = [[10, 0.001], [30, 0.002], [1e9, 0.011]];
const MAX_P = 0.65;
const MIN_P = 0.0005;
// Lifecycle-drift calibration. Horizon knots (hours) at the midpoints of
// the dt buckets used to measure drift; DRIFT_PRIOR[band] is the mean rank
// change of survivors per bucket, measured on the kit archive. The drift
// map is used ONLY to center residuals when calibrating the spread table
// below (it is not the forecast mean: see the header comment). Runtime
// calibration re-estimates each cell from the input history's read pairs
// and shrinks toward the prior with PRIOR_W pseudo-pairs, so thin
// histories fall back to the priors and dense (recorder) histories adapt.
const DRIFT_KNOTS = [3, 9, 18, 36, 72];
const DRIFT_EDGES = [6, 12, 24, 48, 1e9];
const DRIFT_PRIOR: Array<[number, number[]]> = [
  [10, [0.2, 0.8, 2.2, 6.5, 13.4]],
  [20, [0.3, 1.6, 4.0, 9.1, 17.3]],
  [40, [0.2, 1.6, 4.0, 7.9, 14.8]],
  [1e9, [-0.5, -1.2, -2.5, -4.8, -8.1]],
];
const PRIOR_W = 200;
// Residual-spread priors: SD of the rank residual around the drift-map
// mean, per band per dt bucket, measured on the kit archive. The
// conditional distribution is far wider at long horizons than a random
// walk with hourly SD suggests (top band: SD 1.3 at ~3h but 17 at ~72h),
// because games fall at very different rates. Runtime calibration
// re-estimates each cell from the input history; the table is scaled by
// 0.5 at use, per the backtest sweep.
const SIGMA_PRIOR: Array<[number, number[]]> = [
  [10, [1.3, 3.1, 6.3, 13.0, 17.4]],
  [20, [2.5, 5.4, 9.6, 14.8, 20.3]],
  [40, [4.1, 8.9, 13.6, 18.8, 24.4]],
  [1e9, [5.8, 10.9, 16.5, 22.5, 26.7]],
];

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
  hazard: number;
  absentHrs: number;
  placeP: number[];
}

// Calibrate the lifecycle drift map from the input history: for every pair
// of reads, every game present in both contributes its rank change to the
// cell (start band, dt bucket). Cells shrink toward the priors.
const driftTable: number[][] = DRIFT_PRIOR.map(([, row]) => [...row]);
{
  const ordered = [...snaps].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const rankMaps = ordered.map((s) => new Map((s.up_and_coming ?? []).map((r) => [r.universe_id, r.rank])));
  const sums: number[][] = DRIFT_PRIOR.map(() => DRIFT_KNOTS.map(() => 0));
  const counts: number[][] = DRIFT_PRIOR.map(() => DRIFT_KNOTS.map(() => 0));
  const bandIdx = (rank: number): number => {
    for (let i = 0; i < DRIFT_PRIOR.length; i++) if (rank <= DRIFT_PRIOR[i]![0]) return i;
    return DRIFT_PRIOR.length - 1;
  };
  for (let i = 0; i < ordered.length; i++) {
    for (let j = i + 1; j < ordered.length; j++) {
      const dtHrs = (Date.parse(ordered[j]!.at) - Date.parse(ordered[i]!.at)) / HOUR;
      if (dtHrs <= 0 || dtHrs > 96) continue;
      let bucket = 0;
      while (bucket < DRIFT_EDGES.length - 1 && dtHrs >= DRIFT_EDGES[bucket]!) bucket++;
      const later = rankMaps[j]!;
      for (const [id, rank] of rankMaps[i]!) {
        const next = later.get(id);
        if (next === undefined) continue;
        const b = bandIdx(rank);
        sums[b]![bucket]! += next - rank;
        counts[b]![bucket]! += 1;
      }
    }
  }
  for (let b = 0; b < DRIFT_PRIOR.length; b++) {
    for (let k = 0; k < DRIFT_KNOTS.length; k++) {
      const n = counts[b]![k]!;
      driftTable[b]![k] = (sums[b]![k]! + PRIOR_W * DRIFT_PRIOR[b]![1][k]!) / (n + PRIOR_W);
    }
  }
}

function driftAt(rank: number, h: number): number {
  let row = driftTable[driftTable.length - 1]!;
  for (let i = 0; i < DRIFT_PRIOR.length; i++) {
    if (rank <= DRIFT_PRIOR[i]![0]) { row = driftTable[i]!; break; }
  }
  // Piecewise-linear in h through (0, 0) and the knots; flat past 72h.
  let prevH = 0;
  let prevV = 0;
  for (let k = 0; k < DRIFT_KNOTS.length; k++) {
    const kh = DRIFT_KNOTS[k]!;
    if (h <= kh) {
      const f = (h - prevH) / (kh - prevH);
      return prevV + f * (row[k]! - prevV);
    }
    prevH = kh;
    prevV = row[k]!;
  }
  return row[row.length - 1]!;
}

// Calibrate the residual-spread table the same way: squared residuals
// around the drift-map prediction, per (band, dt bucket), shrunk toward
// the priors. Needs the drift table first, so this is a second pass.
const sigmaTable: number[][] = SIGMA_PRIOR.map(([, row]) => [...row]);
{
  const ordered = [...snaps].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const rankMaps = ordered.map((s) => new Map((s.up_and_coming ?? []).map((r) => [r.universe_id, r.rank])));
  const sqSums: number[][] = SIGMA_PRIOR.map(() => DRIFT_KNOTS.map(() => 0));
  const counts: number[][] = SIGMA_PRIOR.map(() => DRIFT_KNOTS.map(() => 0));
  const bandIdx = (rank: number): number => {
    for (let i = 0; i < SIGMA_PRIOR.length; i++) if (rank <= SIGMA_PRIOR[i]![0]) return i;
    return SIGMA_PRIOR.length - 1;
  };
  for (let i = 0; i < ordered.length; i++) {
    for (let j = i + 1; j < ordered.length; j++) {
      const dtHrs = (Date.parse(ordered[j]!.at) - Date.parse(ordered[i]!.at)) / HOUR;
      if (dtHrs <= 0 || dtHrs > 96) continue;
      let bucket = 0;
      while (bucket < DRIFT_EDGES.length - 1 && dtHrs >= DRIFT_EDGES[bucket]!) bucket++;
      const later = rankMaps[j]!;
      for (const [id, rank] of rankMaps[i]!) {
        const next = later.get(id);
        if (next === undefined) continue;
        const resid = next - (rank + driftAt(rank, dtHrs));
        const b = bandIdx(rank);
        sqSums[b]![bucket]! += resid * resid;
        counts[b]![bucket]! += 1;
      }
    }
  }
  for (let b = 0; b < SIGMA_PRIOR.length; b++) {
    for (let k = 0; k < DRIFT_KNOTS.length; k++) {
      const n = counts[b]![k]!;
      const emp = n > 0 ? Math.sqrt(sqSums[b]![k]! / n) : SIGMA_PRIOR[b]![1][k]!;
      sigmaTable[b]![k] = (n * emp + PRIOR_W * SIGMA_PRIOR[b]![1][k]!) / (n + PRIOR_W);
    }
  }
}

function sigmaAt(rank: number, h: number): number {
  let row = sigmaTable[sigmaTable.length - 1]!;
  for (let i = 0; i < SIGMA_PRIOR.length; i++) {
    if (rank <= SIGMA_PRIOR[i]![0]) { row = sigmaTable[i]!; break; }
  }
  // Piecewise-linear in h through (0, 0.9) and the knots; flat past 72h.
  let prevH = 0;
  let prevV = 0.9;
  for (let k = 0; k < DRIFT_KNOTS.length; k++) {
    const kh = DRIFT_KNOTS[k]!;
    if (h <= kh) {
      const f = (h - prevH) / (kh - prevH);
      return Math.max(0.9, prevV + f * (row[k]! - prevV));
    }
    prevH = kh;
    prevV = row[k]!;
  }
  return Math.max(0.9, row[row.length - 1]!);
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
  // are noisy, but backtests on held-out splits showed the unshrunk recent
  // trend (clamped at 0.6 ranks/hour), extrapolated linearly, beat shrunk,
  // saturating and regression-slope versions, so no shrinkage is applied
  // beyond the clamp.
  let velocity = 0;
  const winStart = endMs - 48 * HOUR;
  const early = obs.find((o) => o.t >= winStart) ?? obs[0]!;
  const dtHrs = (last.t - early.t) / HOUR;
  if (dtHrs >= 6) velocity = clamp((last.rank - early.rank) / dtHrs, -0.6, 0.6);
  // Volatility: the calibrated residual-spread table (sigmaAt, scaled by
  // 0.5 at use), which already includes each band's typical movement;
  // per-game own-volatility estimates from sparse reads were too noisy
  // to help.
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
      // Mean: anchor plus the game's own recent trend, extrapolated. A
      // drift-map mean was tried and rejected on backtest evidence (see the
      // header comment).
      const mu = clamp(anchor + velocity * h + 0.1 * absentHrs, 0.5, 130);
      // Spread: calibrated residual SD table, scaled by 0.5 (sweep winner).
      const sigma = 0.5 * sigmaAt(last.rank, h);
      const surv = Math.exp(-hazard * h) * Math.exp(-0.02 * absentHrs);
      acc += surv * (normCdf((k + 0.5 - mu) / sigma) - normCdf((k - 0.5 - mu) / sigma));
    }
    placeP.push(acc / HORIZONS.length);
  }
  forecasts.push({ id: g.id, anchor, velocity, hazard, absentHrs, placeP });
}

// Per-place normalisation: scale the named rows to exactly 1 - RESERVE.
// The unnamed share is the reserve, split over chart games not named here.
// Scaling up as well as down: on held-out splits the actual occupant is
// named ~88% of the time, and the unnamed share per game (~0.005) is far
// below the named probabilities, so leaving honest shortfall unnamed
// wastes mass.
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
  const scale = sum > 0 ? target / sum : 1;
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
