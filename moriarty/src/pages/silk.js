import { useState, useMemo, useCallback, useEffect } from "react";
import Head from "next/head";
import dynamic from "next/dynamic";
import {
  STATIONS,
  nextHops,
  nextHopsLive,
  isSettled,
  residualDistance,
  resolveStation,
  buyLink,
} from "@/silk/network";

// Map touches `window` (Leaflet) — load client-side only.
const SilkMap = dynamic(() => import("@/silk/SilkMap"), { ssr: false });

// ---- small helpers ----
function haversineFallback([lat1, lon1], [lat2, lon2]) {
  const R = 6371, toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// ---- small presentational helpers ----
const TONE = {
  fast: { bg: "#e7f0ff", fg: "#1f6feb", label: "fast" },
  mid: { bg: "#fff3e6", fg: "#b45309", label: "" },
  slow: { bg: "#eef2f4", fg: "#556270", label: "local" },
};

function HopCard({ hop, currentId, onTake }) {
  const tone = TONE[hop.tone] || TONE.mid;
  return (
    <div
      style={{
        border: "1px solid #e5e9f0",
        borderRadius: 14,
        padding: "12px 14px",
        marginBottom: 10,
        background: "#fff",
        boxShadow: "0 1px 2px rgba(16,24,40,0.04)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              fontSize: 12,
              fontWeight: 700,
              color: tone.fg,
              background: tone.bg,
              padding: "2px 8px",
              borderRadius: 999,
            }}
          >
            {hop.clsLabel}
          </span>
          <span style={{ fontWeight: 600, color: "#111827" }}>→ {hop.arriveName}</span>
        </div>
        <span style={{ fontSize: 13, color: "#6b7280" }}>{hop.time}</span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginTop: 10,
        }}
      >
        <div style={{ fontSize: 12.5, color: "#6b7280" }}>
          <span style={{ color: "#16a34a", fontWeight: 600 }}>
            −{Math.round(hop.progressKm)} km
          </span>{" "}
          toward · {Math.round(hop.remainingKm)} km left
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <a
            href={buyLink(hop, currentId)}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "#1f6feb",
              textDecoration: "none",
              border: "1px solid #cfe0ff",
              borderRadius: 10,
              padding: "5px 10px",
              background: "#f5f9ff",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            €{hop.price} · buy
          </a>
          <button
            onClick={() => onTake(hop)}
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "#fff",
              background: "#111827",
              border: "none",
              borderRadius: 10,
              padding: "6px 12px",
              cursor: "pointer",
            }}
          >
            board
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Silk() {
  const stationList = useMemo(() => Object.values(STATIONS), []);
  const [targetId, setTargetId] = useState("berlin");
  const [currentId, setCurrentId] = useState("nuremberg");
  const [trail, setTrail] = useState(["nuremberg"]);
  const [permissive, setPermissive] = useState(false);
  const [live, setLive] = useState(false);           // live vs demo data
  const [hops, setHops] = useState([]);
  const [status, setStatus] = useState("demo");       // "demo" | "live" | "loading" | "fallback"

  const settled = isSettled(currentId, targetId);
  const remaining = useMemo(() => {
    const cur = resolveStation(currentId), tgt = resolveStation(targetId);
    if (!cur || !tgt || !cur.coord || !tgt.coord) return 0;
    return residualDistance(currentId, targetId) || haversineFallback(cur.coord, tgt.coord);
  }, [currentId, targetId]);

  // Recompute hops whenever position/target/mode changes.
  useEffect(() => {
    let cancelled = false;
    if (settled) {
      setHops([]);
      return;
    }
    if (!live) {
      setHops(nextHops(currentId, targetId, { permissive }));
      setStatus("demo");
      return;
    }
    // live mode: fetch, fall back to demo on failure
    setStatus("loading");
    nextHopsLive(currentId, targetId, { permissive }).then((res) => {
      if (cancelled) return;
      if (res.ok && res.hops.length) {
        setHops(res.hops);
        setStatus("live");
      } else {
        // graceful fallback: show demo data, flag it
        setHops(nextHops(currentId, targetId, { permissive }));
        setStatus("fallback");
      }
    });
    return () => { cancelled = true; };
  }, [currentId, targetId, permissive, live, settled]);

  const take = useCallback((hop) => {
    setCurrentId(hop.arriveId);
    setTrail((t) => [...t, hop.arriveId]);
  }, []);

  const restart = useCallback((startId, tgtId) => {
    setCurrentId(startId);
    setTargetId(tgtId);
    setTrail([startId]);
  }, []);

  return (
    <>
      <Head>
        <title>Silk</title>
        <meta name="description" content="Directional rail journey companion" />
      </Head>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          padding: "16px 12px 40px",
          background: "#f3f5f8",
          minHeight: "100vh",
        }}
      >
        {/* phone frame */}
        <div
          style={{
            width: "100%",
            maxWidth: 420,
            background: "#f7f9fc",
            borderRadius: 24,
            overflow: "hidden",
            boxShadow: "0 8px 30px rgba(16,24,40,0.12)",
            border: "1px solid #e5e9f0",
          }}
        >
          {/* header */}
          <div style={{ padding: "16px 18px 10px", background: "#fff", borderBottom: "1px solid #eef1f5" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: -0.5, color: "#111827" }}>
                Silk
              </span>
              <button
                onClick={() => setLive((v) => !v)}
                style={{
                  display: "flex", alignItems: "center", gap: 6,
                  fontSize: 12, fontWeight: 700, cursor: "pointer",
                  border: "1px solid #e5e9f0", borderRadius: 999,
                  padding: "4px 10px", background: "#fff",
                  color: live ? "#16a34a" : "#9aa5b1",
                }}
                title={live ? "Using live DB departures" : "Using demo data"}
              >
                <span
                  style={{
                    width: 8, height: 8, borderRadius: 999,
                    background:
                      status === "live" ? "#16a34a" :
                      status === "loading" ? "#f59e0b" :
                      status === "fallback" ? "#ef4444" : "#cbd5e1",
                  }}
                />
                {status === "loading" ? "…" :
                 status === "live" ? "LIVE" :
                 status === "fallback" ? "live off" : "DEMO"}
              </button>
            </div>
            {status === "fallback" && (
              <div style={{ fontSize: 11, color: "#ef4444", marginTop: 4 }}>
                Live DB feed unreachable — showing demo departures.
              </div>
            )}

            {/* destination + origin pickers */}
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <label style={{ flex: 1, fontSize: 12, color: "#6b7280" }}>
                From
                <select
                  value={currentId}
                  onChange={(e) => restart(e.target.value, targetId)}
                  style={selectStyle}
                >
                  {stationList.map((s) => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </label>
              <label style={{ flex: 1, fontSize: 12, color: "#6b7280" }}>
                To
                <select
                  value={targetId}
                  onChange={(e) => restart(currentId, e.target.value)}
                  style={selectStyle}
                >
                  {stationList.map((s) => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {/* map */}
          <div style={{ height: 240, background: "#eef1f5" }}>
            <SilkMap currentId={currentId} targetId={targetId} hops={hops} trail={trail} />
          </div>

          {/* status strip */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "12px 18px",
              background: "#fff",
              borderBottom: "1px solid #eef1f5",
            }}
          >
            <div>
              <div style={{ fontSize: 11, color: "#9aa5b1", textTransform: "uppercase", letterSpacing: 0.5 }}>
                You are at
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#111827" }}>
                {resolveStation(currentId)?.name}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 11, color: "#9aa5b1", textTransform: "uppercase", letterSpacing: 0.5 }}>
                → {resolveStation(targetId)?.name}
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: settled ? "#16a34a" : "#111827" }}>
                {settled ? "arrived" : `${Math.round(remaining)} km left`}
              </div>
            </div>
          </div>

          {/* hop list / arrival */}
          <div style={{ padding: "14px 16px 22px" }}>
            {settled ? (
              <div
                style={{
                  textAlign: "center",
                  padding: "26px 16px",
                  background: "#fff",
                  borderRadius: 16,
                  border: "1px solid #e5e9f0",
                }}
              >
                <div style={{ fontSize: 30 }}>✓</div>
                <div style={{ fontWeight: 700, color: "#16a34a", marginTop: 6 }}>
                  You reached {resolveStation(targetId)?.name}
                </div>
                <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>
                  {trail.length - 1} hop{trail.length - 1 === 1 ? "" : "s"}:{" "}
                  {trail.map((id) => resolveStation(id)?.name).join(" → ")}
                </div>
                <button onClick={() => restart(currentId, targetId)} style={pillBtn}>
                  plan another
                </button>
              </div>
            ) : (
              <>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 10,
                  }}
                >
                  <span style={{ fontSize: 13, fontWeight: 700, color: "#374151" }}>
                    Next hops toward {resolveStation(targetId)?.name}
                  </span>
                  <button
                    onClick={() => setPermissive((p) => !p)}
                    style={{
                      fontSize: 11.5,
                      color: permissive ? "#1f6feb" : "#9aa5b1",
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      fontWeight: 600,
                    }}
                  >
                    {permissive ? "showing all" : "show all"}
                  </button>
                </div>

                {hops.length === 0 ? (
                  <div style={{ fontSize: 13, color: "#9aa5b1", padding: "18px 4px" }}>
                    No departures toward your destination from here right now.
                    {!permissive && " Try “show all”."}
                  </div>
                ) : (
                  hops.map((h, i) => (
                    <HopCard key={i} hop={h} currentId={currentId} onTake={take} />
                  ))
                )}
              </>
            )}
          </div>

          {/* footer note */}
          <div style={{ padding: "10px 18px 16px", fontSize: 10.5, color: "#aab2bd", textAlign: "center" }}>
            Silk maps options. Every purchase happens with the operator — Silk holds nothing.
          </div>
        </div>
      </div>
    </>
  );
}

const selectStyle = {
  width: "100%",
  marginTop: 4,
  padding: "8px 10px",
  borderRadius: 10,
  border: "1px solid #e5e9f0",
  background: "#fff",
  fontSize: 14,
  color: "#111827",
};

const pillBtn = {
  marginTop: 14,
  fontSize: 13,
  fontWeight: 700,
  color: "#fff",
  background: "#111827",
  border: "none",
  borderRadius: 12,
  padding: "8px 16px",
  cursor: "pointer",
};
