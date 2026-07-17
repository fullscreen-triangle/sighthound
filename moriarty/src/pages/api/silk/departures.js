// Silk — live departures API route.
//
// The browser calls THIS endpoint (same-origin, no CORS worry, no keys exposed);
// this server-side handler calls the public db-rest API
// (https://v6.db.transport.rest) and returns a normalized departure list.
//
// Why server-side: (1) keeps the app in control of caching and rate-limiting,
// (2) if a key is ever needed it stays here, (3) db-rest occasionally IP-blocks
// data-center IPs with 503 — when that happens we return {ok:false} and the
// client falls back to demo data, so Silk never shows a broken screen.
//
// Request:  GET /api/silk/departures?ibnr=8000284&duration=60
// Response: { ok: true, source: "db-rest", departures: [ ...normalized... ] }
//        or { ok: false, source: "db-rest", error: "...", status: 503 }

const DB_REST = "https://v6.db.transport.rest";

// tiny in-memory cache (per server instance). key -> { at, data }
const cache = new Map();
const TTL_MS = 30_000; // 30s: departures don't change faster than this matters

// Map db-rest line.product -> Silk class code (matches CLASSES in network.js).
function productToClass(product, lineName = "") {
  switch (product) {
    case "nationalExpress": return "ICE"; // ICE / ECE
    case "national":        return "IC";  // IC / EC
    case "regionalExpress":
    case "regional":        return lineName.startsWith("RB") ? "RB" : "RE";
    case "suburban":        return "SB";  // S-Bahn
    case "subway":          return "SB";  // U-Bahn -> treat as local
    case "tram":            return "SB";
    case "bus":             return "SB";
    default:                return "RE";
  }
}

function hhmm(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
}

// Normalize one db-rest departure into Silk's shape. The "next stop" comes from
// stopovers[] (the first stopover after this station) when available, else falls
// back to the final direction/headsign.
function normalize(dep, originIbnr) {
  const line = dep.line || {};
  const cls = productToClass(line.product, line.name || "");

  // find the immediate next stop from stopovers
  let nextStop = null;
  if (Array.isArray(dep.stopovers) && dep.stopovers.length) {
    const idx = dep.stopovers.findIndex(
      (s) => s.stop && String(s.stop.id) === String(originIbnr)
    );
    // the stop right after our origin
    const after = idx >= 0 ? dep.stopovers[idx + 1] : dep.stopovers[0];
    if (after && after.stop) {
      nextStop = {
        ibnr: String(after.stop.id),
        name: after.stop.name,
        coord: after.stop.location
          ? [after.stop.location.latitude, after.stop.location.longitude]
          : null,
      };
    }
  }

  return {
    tripId: dep.tripId,
    line: line.name || cls,
    cls,
    product: line.product,
    direction: dep.direction || (dep.destination && dep.destination.name) || "",
    when: dep.when,
    plannedWhen: dep.plannedWhen,
    time: hhmm(dep.when || dep.plannedWhen),
    delayMin: typeof dep.delay === "number" ? Math.round(dep.delay / 60) : null,
    platform: dep.platform || dep.plannedPlatform || null,
    nextStop, // {ibnr, name, coord} or null
  };
}

export default async function handler(req, res) {
  const ibnr = String(req.query.ibnr || "").trim();
  const duration = Math.min(parseInt(req.query.duration || "60", 10) || 60, 120);
  if (!/^\d{5,9}$/.test(ibnr)) {
    return res.status(400).json({ ok: false, error: "invalid or missing ibnr" });
  }

  const key = `${ibnr}:${duration}`;
  const cached = cache.get(key);
  if (cached && Date.now() - cached.at < TTL_MS) {
    return res.status(200).json({ ...cached.data, cached: true });
  }

  const url =
    `${DB_REST}/stops/${ibnr}/departures` +
    `?duration=${duration}&results=40&stopovers=true&remarks=false`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const upstream = await fetch(url, {
      headers: { "User-Agent": "silk-app (rail journey companion; contact via app)" },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!upstream.ok) {
      // 503 = DB IP-block / throttle; signal client to fall back to demo data.
      return res.status(200).json({
        ok: false,
        source: "db-rest",
        status: upstream.status,
        error: `upstream returned ${upstream.status}`,
      });
    }

    const json = await upstream.json();
    const list = Array.isArray(json) ? json : json.departures || [];
    const departures = list
      .map((d) => normalize(d, ibnr))
      .filter((d) => d.nextStop); // only usable if we know the next stop

    const data = { ok: true, source: "db-rest", ibnr, departures };
    cache.set(key, { at: Date.now(), data });
    return res.status(200).json(data);
  } catch (e) {
    return res.status(200).json({
      ok: false,
      source: "db-rest",
      error: e.name === "AbortError" ? "upstream timeout" : String(e.message || e),
    });
  }
}
