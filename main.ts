import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

type RankRow = { rank: number; universe_id: number };
type Snap = { at: string; up_and_coming: RankRow[]; details?: Record<string, { created?: string | null } | null> | null };
const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as { places: number; output: string; history_until?: string };
const snapshots = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n').map((x) => JSON.parse(x) as Snap);
const hour = 3600000;
const end = Date.parse(task.history_until ?? snapshots[snapshots.length - 1]!.at);
type Observation = { t: number; rank: number };
type RecordGame = { id: number; history: Observation[]; created: number | null };
const games = new Map<number, RecordGame>();
for (const snap of snapshots) {
  const t = Date.parse(snap.at);
  for (const row of snap.up_and_coming ?? []) {
    let g = games.get(row.universe_id);
    if (!g) { g = { id: row.universe_id, history: [], created: null }; games.set(row.universe_id, g); }
    g.history.push({ t, rank: row.rank });
  }
  if (snap.details) for (const [idText, d] of Object.entries(snap.details)) {
    const g = games.get(Number(idText));
    if (g && d?.created && g.created === null) g.created = Date.parse(d.created);
  }
}

const horizons = [4, 8, 16, 24, 36, 48, 60, 72];
const SPREAD_SCALE = 0.35;
const NAMED_SHARE = 0.8;
const spread = [[1.3, 3.1, 6.3, 13, 17.4], [2.5, 5.4, 9.6, 14.8, 20.3], [4.1, 8.9, 13.6, 18.8, 24.4], [5.8, 10.9, 16.5, 22.5, 26.7]];
const edges = [6, 12, 24, 48];
const knots = [3, 9, 18, 36, 72];
function band(rank: number): number { return rank <= 10 ? 0 : rank <= 20 ? 1 : rank <= 40 ? 2 : 3; }
function sigma(rank: number, h: number): number {
  const values = spread[band(rank)]!;
  let priorH = 0, priorV = 0.9;
  for (let i = 0; i < edges.length + 1; i++) {
    const nextH = i < edges.length ? knots[i]! : 72;
    const nextV = values[i]!;
    if (h <= nextH) return Math.max(0.9, priorV + (h - priorH) / (nextH - priorH) * (nextV - priorV));
    priorH = nextH; priorV = nextV;
  }
  return values[4]!;
}
function cdf(x: number): number {
  const sign = x < 0 ? -1 : 1, z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-z * z);
  return 0.5 * (1 + sign * erf);
}
const forecast: { id: number; center: number; mass: number[] }[] = [];
for (const g of games.values()) {
  const obs = g.history;
  const last = obs[obs.length - 1]!;
  if ((end - last.t) / hour > 96) continue;
  let weighted = 0, weights = 0;
  for (const o of obs) { const w = Math.exp(-(end - o.t) / (36 * hour)); weighted += w * o.rank; weights += w; }
  const center = 0.65 * last.rank + 0.35 * weighted / weights;
  const cutoff = end - 48 * hour;
  const early = obs.find((o) => o.t >= cutoff) ?? obs[0]!;
  const dt = (last.t - early.t) / hour;
  const velocity = dt >= 6 ? Math.max(-0.6, Math.min(0.6, (last.rank - early.rank) / dt)) : 0;
  const absent = Math.max(0, (end - last.t) / hour);
  let hazard = [0.001, 0.002, 0.011][last.rank <= 10 ? 0 : last.rank <= 30 ? 1 : 2]!;
  const age = g.created === null ? (end - obs[0]!.t) / hour : (end - g.created) / hour;
  if (age > 480) hazard *= 1 + (g.created === null ? 0.5 : 2.5) * (age - 480) / 192;
  const mass: number[] = [];
  for (let place = 1; place <= task.places; place++) {
    let p = 0;
    for (const h of horizons) {
      const mu = Math.max(0.5, Math.min(130, center + velocity * h + 0.1 * absent));
      const sd = SPREAD_SCALE * sigma(last.rank, h);
      const occupancy = cdf((place + 0.5 - mu) / sd) - cdf((place - 0.5 - mu) / sd);
      p += occupancy * Math.exp(-hazard * h - 0.02 * absent);
    }
    mass.push(p / horizons.length);
  }
  forecast.push({ id: g.id, center, mass });
}
const probabilities: { universe_id: number; place: number; p: number }[] = [];
for (let place = 1; place <= task.places; place++) {
  const candidates = forecast.map((g) => ({ id: g.id, p: g.mass[place - 1]! })).filter((x) => x.p >= 0.0005).sort((a, b) => b.p - a.p || a.id - b.id).slice(0, 100);
  const total = candidates.reduce((s, x) => s + x.p, 0);
  for (const x of candidates) {
    const p = Math.min(0.65, x.p * NAMED_SHARE / total);
    if (p >= 0.0005) probabilities.push({ universe_id: x.id, place, p });
  }
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);
const ranked = forecast.map((g) => ({ id: g.id, score: g.mass.reduce((a, b) => a + b, 0), center: g.center })).sort((a, b) => b.score - a.score || a.center - b.center || a.id - b.id);
const top10 = ranked.slice(0, task.places).map((x) => x.id);
if (top10.length < task.places) for (const row of [...(snapshots.at(-1)?.up_and_coming ?? [])].sort((a, b) => a.rank - b.rank)) if (!top10.includes(row.universe_id) && top10.length < task.places) top10.push(row.universe_id);
mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, task.output), JSON.stringify({ probabilities, top10 }) + '\n');
