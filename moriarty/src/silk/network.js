// Silk — network model and next-hop logic.
//
// Phase 1: a small hand-built German rail network with fake departures, plus
// the "next hops toward X" gradient logic. The logic is written the way the
// paper's model defines it (items descend a residual-distance gradient toward a
// private target), so when we later swap fake departures for live operator
// feeds, only the data source changes — not this logic.
//
// A "hop" = one departing service from the current station whose next stop
// reduces the straight-line distance to the destination.

// --- Stations: id, display name, coordinates [lat, lon], and DB IBNR (evaNr).
// The `ibnr` is the stable public station identifier used by the live DB API
// (v6.db.transport.rest /stops/{ibnr}/departures). Precomputing them lets us
// skip a /locations lookup per station. Values are well-known DB EVA numbers.
export const STATIONS = {
  nuremberg:   { id: "nuremberg",   name: "Nürnberg Hbf",      coord: [49.4459, 11.0829], ibnr: "8000284" },
  bamberg:     { id: "bamberg",     name: "Bamberg",           coord: [49.9010, 10.8859], ibnr: "8000025" },
  erfurt:      { id: "erfurt",      name: "Erfurt Hbf",        coord: [50.9725, 11.0383], ibnr: "8010101" },
  halle:       { id: "halle",       name: "Halle (Saale) Hbf", coord: [51.4776, 11.9874], ibnr: "8010159" },
  leipzig:     { id: "leipzig",     name: "Leipzig Hbf",       coord: [51.3454, 12.3812], ibnr: "8010205" },
  bitterfeld:  { id: "bitterfeld",  name: "Bitterfeld",        coord: [51.6231, 12.3164], ibnr: "8010036" },
  wittenberg:  { id: "wittenberg",  name: "Lutherstadt Wittenberg", coord: [51.8663, 12.6602], ibnr: "8010222" },
  berlin:      { id: "berlin",      name: "Berlin Hbf",        coord: [52.5250, 13.3694], ibnr: "8011160" },
  jena:        { id: "jena",        name: "Jena Paradies",     coord: [50.9200, 11.5900], ibnr: "8010184" },
  munich:      { id: "munich",      name: "München Hbf",       coord: [48.1402, 11.5586], ibnr: "8000261" },
  ingolstadt:  { id: "ingolstadt",  name: "Ingolstadt Hbf",    coord: [48.7447, 11.4370], ibnr: "8000183" },
};

// Reverse lookup: DB IBNR -> our station id (for matching live-departure next stops).
export const IBNR_TO_ID = Object.fromEntries(
  Object.values(STATIONS).filter((s) => s.ibnr).map((s) => [s.ibnr, s.id])
);

// Speed classes: label + a rough top speed (km/h), used only for annotation/flavour.
export const CLASSES = {
  ICE: { label: "ICE",  kmh: 300, tone: "fast" },
  IC:  { label: "IC",   kmh: 200, tone: "mid" },
  RE:  { label: "RE",   kmh: 160, tone: "mid" },
  RB:  { label: "RB",   kmh: 140, tone: "slow" },
  SB:  { label: "S-Bahn", kmh: 120, tone: "slow" },
};

// --- Fake scheduled departures (Phase 1). Each: from, to, class, dep "HH:MM", price € ---
// These stand in for what a live operator feed will later provide.
export const DEPARTURES = [
  // out of Nuremberg
  { from: "nuremberg", to: "erfurt",     cls: "ICE", dep: "12:04", price: 39 },
  { from: "nuremberg", to: "bamberg",    cls: "RE",  dep: "12:11", price: 13 },
  { from: "nuremberg", to: "ingolstadt", cls: "RB",  dep: "12:09", price: 15 },
  { from: "nuremberg", to: "munich",     cls: "ICE", dep: "12:20", price: 55 },
  // out of Bamberg
  { from: "bamberg",   to: "erfurt",     cls: "IC",  dep: "12:47", price: 22 },
  { from: "bamberg",   to: "jena",       cls: "RE",  dep: "12:39", price: 18 },
  // out of Erfurt
  { from: "erfurt",    to: "halle",      cls: "ICE", dep: "13:02", price: 29 },
  { from: "erfurt",    to: "leipzig",    cls: "ICE", dep: "13:08", price: 31 },
  { from: "erfurt",    to: "jena",       cls: "RB",  dep: "12:55", price: 9  },
  // out of Halle
  { from: "halle",     to: "bitterfeld", cls: "RE",  dep: "13:34", price: 8  },
  { from: "halle",     to: "berlin",     cls: "ICE", dep: "13:40", price: 34 },
  // out of Leipzig
  { from: "leipzig",   to: "bitterfeld", cls: "SB",  dep: "13:29", price: 7  },
  { from: "leipzig",   to: "berlin",     cls: "ICE", dep: "13:45", price: 36 },
  // out of Bitterfeld
  { from: "bitterfeld", to: "wittenberg", cls: "RB", dep: "13:52", price: 6  },
  { from: "bitterfeld", to: "berlin",    cls: "IC",  dep: "14:01", price: 24 },
  // out of Wittenberg
  { from: "wittenberg", to: "berlin",    cls: "RE",  dep: "14:20", price: 16 },
  // out of Jena
  { from: "jena",      to: "leipzig",    cls: "RE",  dep: "13:30", price: 14 },
];

// --- Geography: great-circle (haversine) distance in km between two station ids ---
export function residualDistance(fromId, toId) {
  const a = STATIONS[fromId], b = STATIONS[toId];
  if (!a || !b) return Infinity;
  const R = 6371; // km
  const toRad = (d) => (d * Math.PI) / 180;
  const [lat1, lon1] = a.coord, [lat2, lon2] = b.coord;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// --- The core: next hops toward a target, ranked by target-ward progress ---
//
// A departure from `currentId` qualifies as a hop toward `targetId` if its
// arrival station is strictly closer to the target than the current station.
// This is the paper's greedy target-ward routing (descend the residual-distance
// gradient), applied to the available departures.
//
// Returns [{ dep, arriveId, arriveName, cls, price, time, progressKm, remainingKm }]
// sorted by progressKm descending (most target-ward first).
export function nextHops(currentId, targetId, { permissive = false, minSegmentKm = 0 } = {}) {
  if (!currentId || !targetId || currentId === targetId) return [];
  const here = residualDistance(currentId, targetId);

  const hops = DEPARTURES
    .filter((d) => d.from === currentId)
    .map((d) => {
      const remaining = residualDistance(d.to, targetId);
      const progress = here - remaining; // target-ward reduction (km)
      const segmentKm = residualDistance(d.from, d.to); // physical hop length
      return {
        dep: d,
        arriveId: d.to,
        arriveName: STATIONS[d.to]?.name ?? d.to,
        cls: d.cls,
        clsLabel: CLASSES[d.cls]?.label ?? d.cls,
        tone: CLASSES[d.cls]?.tone ?? "mid",
        price: d.price,
        time: d.dep,
        segmentKm,
        progressKm: progress,
        remainingKm: remaining,
      };
    })
    // directional filter: only hops that actually move toward the target
    .filter((h) => (permissive ? h.progressKm > 0.001 : h.progressKm > 1.0))
    // min-segment constraint (paper: "shortest a segment can be")
    .filter((h) => h.segmentKm >= minSegmentKm)
    .sort((a, b) => b.progressKm - a.progressKm);

  return hops;
}

// Convenience: is a station the target (journey complete)?
export function isSettled(currentId, targetId) {
  return currentId === targetId;
}

// A deep-link stand-in for "buy this leg from the operator".
// Phase 4 will replace this with a real bahn.de / operator checkout URL.
export function buyLink(hop, currentId) {
  const from = encodeURIComponent(STATIONS[currentId]?.name ?? currentId);
  const to = encodeURIComponent(hop.arriveName);
  // bahn.de query deep-link (illustrative; the real integration comes later)
  return `https://www.bahn.de/buchung/fahrplan/suche#sts=true&so=${from}&zo=${to}`;
}

// ---------------------------------------------------------------------------
// LIVE data provider (Phase 4)
// ---------------------------------------------------------------------------
//
// A dynamic station registry: live departures reference stations that may not be
// in our hardcoded STATIONS map. We register them on the fly (from the API's
// name + coordinates) so the gradient can be computed against any real station.

const dynamicStations = {}; // ibnr -> { id, name, coord, ibnr, dynamic:true }

function haversineKm(coordA, coordB) {
  if (!coordA || !coordB) return Infinity;
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const [lat1, lon1] = coordA, [lat2, lon2] = coordB;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// Resolve a station reference (by our id, or by ibnr) to {name, coord}.
export function resolveStation(idOrIbnr) {
  if (STATIONS[idOrIbnr]) return STATIONS[idOrIbnr];
  const byIbnr = Object.values(STATIONS).find((s) => s.ibnr === idOrIbnr);
  if (byIbnr) return byIbnr;
  if (dynamicStations[idOrIbnr]) return dynamicStations[idOrIbnr];
  return null;
}

function registerDynamic(nextStop) {
  if (!nextStop || !nextStop.ibnr) return null;
  const existing = resolveStation(nextStop.ibnr);
  if (existing) return existing;
  const s = {
    id: nextStop.ibnr,
    name: nextStop.name,
    coord: nextStop.coord,
    ibnr: nextStop.ibnr,
    dynamic: true,
  };
  dynamicStations[nextStop.ibnr] = s;
  return s;
}

// Fetch live next-hops toward a target from the Silk API route.
// Returns { ok, source, hops } where hops match the fake nextHops() shape.
// On any failure returns { ok:false } so the caller can fall back to demo data.
export async function nextHopsLive(currentId, targetId, { permissive = false, minSegmentKm = 0 } = {}) {
  const cur = resolveStation(currentId);
  const tgt = resolveStation(targetId);
  if (!cur || !tgt || !cur.ibnr) return { ok: false, hops: [] };

  let resp;
  try {
    const r = await fetch(
      `/api/silk/departures?ibnr=${encodeURIComponent(cur.ibnr)}&duration=90`
    );
    resp = await r.json();
  } catch (e) {
    return { ok: false, hops: [], error: String(e) };
  }
  if (!resp || !resp.ok) return { ok: false, hops: [], error: resp && resp.error, status: resp && resp.status };

  const here = haversineKm(cur.coord, tgt.coord);
  const seen = new Set();
  const hops = [];

  for (const d of resp.departures) {
    const dest = registerDynamic(d.nextStop);
    if (!dest || !dest.coord) continue;
    const remaining = haversineKm(dest.coord, tgt.coord);
    const progress = here - remaining;
    const threshold = permissive ? 0.1 : 1.0;
    if (progress <= threshold) continue;

    const segmentKm = haversineKm(cur.coord, dest.coord);
    if (segmentKm < minSegmentKm) continue; // min-segment constraint

    // de-duplicate: keep the earliest departure per (nextStop, class)
    const dedupeKey = `${dest.id}|${d.cls}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);

    hops.push({
      dep: { from: cur.id, to: dest.id, cls: d.cls, dep: d.time },
      arriveId: dest.id,
      arriveName: dest.name,
      cls: d.cls,
      clsLabel: (CLASSES[d.cls] && CLASSES[d.cls].label) || d.line || d.cls,
      tone: (CLASSES[d.cls] && CLASSES[d.cls].tone) || "mid",
      line: d.line,
      price: null, // live price requires a booking call; not fetched here
      time: d.time,
      delayMin: d.delayMin,
      platform: d.platform,
      direction: d.direction,
      segmentKm,
      progressKm: progress,
      remainingKm: remaining,
    });
  }

  hops.sort((a, b) => b.progressKm - a.progressKm);
  return { ok: true, source: resp.source, hops };
}
