import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

type Observation = { time: number; rank: number };
type GameRecord = { observations: Observation[]; created: number | null };
type Snapshot = {
  at: string;
  up_and_coming?: { rank: number; universe_id: number }[];
  details?: Record<string, { created?: string } | null>;
};

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
const task = JSON.parse(readFileSync(join(inputDir, "task.json"), "utf8"));
const outputFile =
  task.output_file ?? task.outputFile ?? task.file_name ?? task.filename ?? task.output ?? "forecast.json";
const historyText = readFileSync(join(inputDir, "history.ndjson"), "utf8").trimEnd();
const snapshots: Snapshot[] = historyText.split("\n").map((line) => JSON.parse(line));
const historyCutoff = task.history_until;
const end = historyCutoff !== undefined
  ? Date.parse(historyCutoff)
  : Date.parse(snapshots[snapshots.length - 1].at);
const records = new Map<number, GameRecord>();

for (const snapshot of snapshots) {
  const time = Date.parse(snapshot.at);
  for (const row of snapshot.up_and_coming ?? []) {
    let record = records.get(row.universe_id);
    if (!record) {
      record = { observations: [], created: null };
      records.set(row.universe_id, record);
    }
    record.observations.push({ time, rank: row.rank });
  }
  if (snapshot.details) {
    for (const [id, detail] of Object.entries(snapshot.details)) {
      const record = records.get(Number(id));
      if (record && record.created === null && detail?.created) {
        record.created = Date.parse(detail.created);
      }
    }
  }
}

const horizons = [4, 8, 16, 24, 36, 48, 60, 72];
const knots = [3, 9, 18, 36, 72];
const spreads = [
  [1.3, 3.1, 6.3, 13, 17.4],
  [2.5, 5.4, 9.6, 14.8, 20.3],
  [4.1, 8.9, 13.6, 18.8, 24.4],
  [5.8, 10.9, 16.5, 22.5, 26.7],
];
const bandEdges = [6, 12, 24, 48];
const spreadScale = 0.35;
const share = 0.8;
const probabilityFloor = 0.0005;
const probabilityCap = 0.65;

function spread(rank: number, horizon: number): number {
  const band = rank <= 10 ? 0 : rank <= 20 ? 1 : rank <= 40 ? 2 : 3;
  const row = spreads[band];
  let previousHorizon = 0;
  let previousValue = 0.9;
  for (let i = 0; i < 5; i++) {
    const nextHorizon = i < 4 ? knots[i] : 72;
    const nextValue = row[i];
    if (horizon <= nextHorizon) {
      return Math.max(
        0.9,
        previousValue +
          ((horizon - previousHorizon) / (nextHorizon - previousHorizon)) *
            (nextValue - previousValue),
      );
    }
    previousHorizon = nextHorizon;
    previousValue = nextValue;
  }
  return row[4];
}

function normalCdf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const zed = Math.abs(x) / Math.sqrt(2);
  const tee = 1 / (1 + 0.3275911 * zed);
  const polynomial =
    (((((1.061405429 * tee - 1.453152027) * tee + 1.421413741) * tee -
      0.284496736) *
      tee +
      0.254829592) *
      tee);
  const erf = 1 - polynomial * Math.exp(-zed * zed);
  return 0.5 * (1 + sign * erf);
}

const forecasts: { id: number; center: number; masses: number[]; total: number }[] = [];
for (const [id, record] of records) {
  const observations = record.observations;
  const last = observations[observations.length - 1];
  const lastAge = (end - last.time) / 3600000;
  if (lastAge > 96) continue;
  let weightedRank = 0;
  let weightTotal = 0;
  for (const observation of observations) {
    const weight = Math.exp(-(end - observation.time) / 129600000);
    weightedRank += observation.rank * weight;
    weightTotal += weight;
  }
  const center = 0.65 * last.rank + (0.35 * weightedRank) / weightTotal;
  const cutoff = end - 48 * 3600000;
  const early = observations.find((observation) => observation.time >= cutoff) ?? observations[0];
  const span = (last.time - early.time) / 3600000;
  const velocity =
    span >= 6
      ? Math.max(-0.6, Math.min(0.6, (last.rank - early.rank) / span))
      : 0;
  const absence = Math.max(0, lastAge);
  let hazard = last.rank <= 10 ? 0.001 : last.rank <= 30 ? 0.002 : 0.011;
  const age = (end - (record.created ?? observations[0].time)) / 3600000;
  if (age > 480) {
    hazard *= 1 + ((record.created === null ? 0.5 : 2.5) * (age - 480)) / 192;
  }
  const masses: number[] = [];
  for (let place = 1; place <= task.places; place++) {
    let sum = 0;
    for (const horizon of horizons) {
      const mean = Math.max(
        0.5,
        Math.min(130, center + velocity * horizon + 0.1 * absence),
      );
      const standardDeviation = spreadScale * spread(last.rank, horizon);
      const occupancy =
        normalCdf((place + 0.5 - mean) / standardDeviation) -
        normalCdf((place - 0.5 - mean) / standardDeviation);
      sum += occupancy * Math.exp(-hazard * horizon - 0.02 * absence);
    }
    masses.push(sum / horizons.length);
  }
  forecasts.push({
    id,
    center,
    masses,
    total: masses.reduce((sum, value) => sum + value, 0),
  });
}

const probabilities: { universe_id: number; place: number; p: number }[] = [];
for (let place = 1; place <= task.places; place++) {
  const candidates = forecasts
    .map((forecast) => ({ id: forecast.id, mass: forecast.masses[place - 1] }))
    .filter((candidate) => candidate.mass >= probabilityFloor)
    .sort((a, b) => b.mass - a.mass || a.id - b.id)
    .slice(0, 100);
  const total = candidates.reduce((sum, candidate) => sum + candidate.mass, 0);
  for (const candidate of candidates) {
    const probability = Math.min(probabilityCap, (candidate.mass * share) / total);
    if (probability >= probabilityFloor) {
      probabilities.push({ universe_id: candidate.id, place, p: probability });
    }
  }
}
probabilities.sort(
  (a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id,
);

forecasts.sort(
  (a, b) => b.total - a.total || a.center - b.center || a.id - b.id,
);
const top10 = forecasts.slice(0, task.places).map((forecast) => forecast.id);
if (top10.length < task.places) {
  const lastRows = snapshots[snapshots.length - 1].up_and_coming ?? [];
  lastRows.sort((a, b) => a.rank - b.rank);
  for (const row of lastRows) {
    if (!top10.includes(row.universe_id)) top10.push(row.universe_id);
    if (top10.length >= task.places) break;
  }
}

mkdirSync(outputDir, { recursive: true });
writeFileSync(
  join(outputDir, outputFile),
  JSON.stringify({ probabilities, top10 }) + "\n",
);
