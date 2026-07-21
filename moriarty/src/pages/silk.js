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

// deck.gl map touches WebGL/window — client-only.
const SilkGlobe = dynamic(() => import("@/silk/SilkGlobe"), { ssr: false });

// ---------------------------------------------------------------------------
// The Q&A parameters (from silk-structure.md). Each maps to a paper constraint.
// ---------------------------------------------------------------------------
const QUESTIONS = [
  {
    key: "latestArrival",
    prompt: "Latest you want to arrive?",
    hint: "we descend toward your target in time",
    options: [
      { label: "No limit", value: null },
      { label: "In 3 h", value: 180 },
      { label: "In 6 h", value: 360 },
      { label: "Today", value: 720 },
    ],
  },
  {
    key: "maxStops",
    prompt: "Most stops you'll accept?",
    hint: "bounds how many times you change",
    options: [
      { label: "2", value: 2 },
      { label: "4", value: 4 },
      { label: "6", value: 6 },
      { label: "Any", value: 99 },
    ],
  },
  {
    key: "maxWait",
    prompt: "Longest you'll wait, total?",
    hint: "time spent standing in stations",
    options: [
      { label: "10 min", value: 10 },
      { label: "30 min", value: 30 },
      { label: "1 h", value: 60 },
      { label: "Any", value: 999 },
    ],
  },
  {
    key: "minSegment",
    prompt: "Shortest a leg should be?",
    hint: "skip tiny hops; ride longer stretches",
    options: [
      { label: "Any", value: 0 },
      { label: "20 km", value: 20 },
      { label: "50 km", value: 50 },
      { label: "100 km", value: 100 },
    ],
  },
  {
    key: "density",
    prompt: "Preferred crowd level?",
    hint: "quieter carriages vs. fastest",
    options: [
      { label: "Quiet", value: "low" },
      { label: "Balanced", value: "mid" },
      { label: "Don't mind", value: "any" },
    ],
  },
];

// Limited destination set for now (the structure asks to keep it small).
const DESTS = ["berlin", "munich", "leipzig", "erfurt", "nuremberg"];

export default function Silk() {
  // flow stage: home -> search -> params -> journey (walk|vehicle)
  const [stage, setStage] = useState("home");
  const [originId, setOriginId] = useState("nuremberg"); // the glowing dot start
  const [targetId, setTargetId] = useState(null);

  // Q&A answers
  const [params, setParams] = useState({});
  const [qIndex, setQIndex] = useState(0);

  // journey state
  const [currentId, setCurrentId] = useState("nuremberg");
  const [trail, setTrail] = useState(["nuremberg"]);
  const [leg, setLeg] = useState("walk"); // walk -> vehicle
  const [hops, setHops] = useState([]);
  const [live, setLive] = useState(false);
  const [status, setStatus] = useState("demo");

  const settled = targetId ? isSettled(currentId, targetId) : false;

  // geometry helpers for the map
  const posCoord = resolveStation(currentId)?.coord ?? STATIONS[originId].coord;
  const targetCoord = targetId ? resolveStation(targetId)?.coord ?? null : null;
  const trailCoords = useMemo(
    () => trail.map((id) => resolveStation(id)?.coord).filter(Boolean),
    [trail]
  );
  const hopEndpoints = useMemo(
    () => hops.map((h) => ({ coord: resolveStation(h.arriveId)?.coord, tone: h.tone })),
    [hops]
  );

  // recompute hops during the vehicle leg
  useEffect(() => {
    if (stage !== "journey" || leg !== "vehicle" || !targetId || settled) {
      setHops([]);
      return;
    }
    let cancelled = false;
    const opts = { minSegmentKm: params.minSegment || 0 };
    if (!live) {
      setHops(nextHops(currentId, targetId, opts));
      setStatus("demo");
      return;
    }
    setStatus("loading");
    nextHopsLive(currentId, targetId, opts).then((res) => {
      if (cancelled) return;
      if (res.ok && res.hops.length) { setHops(res.hops); setStatus("live"); }
      else { setHops(nextHops(currentId, targetId, opts)); setStatus("fallback"); }
    });
    return () => { cancelled = true; };
  }, [stage, leg, currentId, targetId, live, settled, params.minSegment]);

  // ---- actions ----
  const chooseDest = useCallback((id) => {
    setTargetId(id);
    setParams({});
    setQIndex(0);
    setStage("params");
  }, []);

  const answer = useCallback((val) => {
    setParams((p) => ({ ...p, [QUESTIONS[qIndex].key]: val }));
    if (qIndex + 1 < QUESTIONS.length) {
      setQIndex((i) => i + 1);
    } else {
      // done with Q&A -> start the journey at the walking leg
      setCurrentId(originId);
      setTrail([originId]);
      setLeg("walk");
      setStage("journey");
    }
  }, [qIndex, originId]);

  const take = useCallback((hop) => {
    setCurrentId(hop.arriveId);
    setTrail((t) => [...t, hop.arriveId]);
  }, []);

  const reset = useCallback(() => {
    setStage("home");
    setTargetId(null);
    setParams({});
    setQIndex(0);
    setCurrentId(originId);
    setTrail([originId]);
    setLeg("walk");
  }, [originId]);

  const remainingKm = targetId ? residualDistance(currentId, targetId) : 0;
  const stopsUsed = Math.max(trail.length - 1, 0);

  return (
    <>
      <Head>
        <title>Silk</title>
        <meta name="description" content="Directional rail journey companion" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
      </Head>

      {/* full-bleed map surface */}
      <div style={{ position: "fixed", inset: 0, background: "#0b0f17", overflow: "hidden" }}>
        {/* map sits at the base layer (z-index 0); all overlays paint above it
            and set their own pointer-events, so map panning never steals a click
            meant for the search/Q&A/buttons. */}
        <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
          <SilkGlobe
            position={posCoord}
            target={targetCoord}
            hops={stage === "journey" && leg === "vehicle" ? hopEndpoints : []}
            trail={trailCoords}
            pitch={stage === "home" ? 45 : 55}
            zoom={18}
            interactive={stage === "home" || stage === "journey"}
          />
        </div>

        {/* ---------- top bar / search affordance ---------- */}
        <TopBar
          stage={stage}
          live={live}
          status={status}
          onLive={() => setLive((v) => !v)}
          onSearch={() => setStage("search")}
          onReset={reset}
          originName={STATIONS[originId].name}
          targetName={targetId ? resolveStation(targetId)?.name : null}
        />

        {/* ---------- search overlay ---------- */}
        {stage === "search" && (
          <SearchOverlay
            onPick={chooseDest}
            onClose={() => setStage("home")}
          />
        )}

        {/* ---------- Q&A parameter sequence (no panel background) ---------- */}
        {stage === "params" && (
          <QAOverlay
            q={QUESTIONS[qIndex]}
            index={qIndex}
            total={QUESTIONS.length}
            targetName={resolveStation(targetId)?.name}
            onAnswer={answer}
          />
        )}

        {/* ---------- journey: walking then vehicle ---------- */}
        {stage === "journey" && (
          <JourneyOverlay
            leg={leg}
            settled={settled}
            currentName={resolveStation(currentId)?.name}
            targetName={resolveStation(targetId)?.name}
            remainingKm={remainingKm}
            stopsUsed={stopsUsed}
            maxStops={params.maxStops}
            hops={hops}
            status={status}
            onStartVehicle={() => setLeg("vehicle")}
            onTake={take}
            onReset={reset}
            currentId={currentId}
          />
        )}
      </div>
    </>
  );
}

// ===========================================================================
// Sub-components
// ===========================================================================

function TopBar({ stage, live, status, onLive, onSearch, onReset, originName, targetName }) {
  return (
    <div
      style={{
        position: "absolute", top: 0, left: 0, right: 0,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "16px 18px", zIndex: 20, pointerEvents: "none",
      }}
    >
      <div style={{ pointerEvents: "auto" }}>
        <div style={{ fontSize: 20, fontWeight: 800, color: "#eaf6ff", letterSpacing: -0.4,
                      textShadow: "0 1px 8px rgba(0,0,0,0.6)" }}>
          Silk
        </div>
        {stage !== "home" && targetName && (
          <div style={{ fontSize: 12, color: "#9fd8ff", textShadow: "0 1px 6px rgba(0,0,0,0.6)" }}>
            → {targetName}
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8, pointerEvents: "auto" }}>
        {(stage === "journey") && (
          <button onClick={onLive} style={pillStyle(live ? "#16341f" : "rgba(255,255,255,0.08)")}>
            <span style={{
              width: 7, height: 7, borderRadius: 999, marginRight: 6, display: "inline-block",
              background: status === "live" ? "#22c55e" : status === "loading" ? "#f59e0b" :
                          status === "fallback" ? "#ef4444" : "#94a3b8",
            }} />
            {status === "loading" ? "…" : status === "live" ? "LIVE" :
             status === "fallback" ? "demo" : "DEMO"}
          </button>
        )}
        {stage === "home" ? (
          <button onClick={onSearch} style={pillStyle("rgba(255,255,255,0.12)")}>
            ⌕ destination
          </button>
        ) : (
          <button onClick={onReset} style={pillStyle("rgba(255,255,255,0.08)")}>
            reset
          </button>
        )}
      </div>
    </div>
  );
}

function SearchOverlay({ onPick, onClose }) {
  const [q, setQ] = useState("");
  const results = DESTS
    .map((id) => STATIONS[id])
    .filter((s) => s.name.toLowerCase().includes(q.toLowerCase()));
  return (
    <div
      style={{
        position: "absolute", inset: 0, zIndex: 30,
        background: "rgba(6,10,17,0.72)", backdropFilter: "blur(3px)",
        display: "flex", flexDirection: "column", padding: "70px 20px 20px",
      }}
      onClick={onClose}
    >
      <input
        autoFocus
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        placeholder="Where to?"
        style={{
          width: "100%", fontSize: 20, fontWeight: 600, color: "#eaf6ff",
          background: "rgba(255,255,255,0.06)", border: "1px solid rgba(159,216,255,0.3)",
          borderRadius: 14, padding: "14px 16px", outline: "none",
        }}
      />
      <div style={{ marginTop: 14 }}>
        {results.map((s) => (
          <button
            key={s.id}
            onClick={(e) => { e.stopPropagation(); onPick(s.id); }}
            style={{
              display: "block", width: "100%", textAlign: "left",
              fontSize: 17, fontWeight: 600, color: "#dbeeff",
              background: "transparent", border: "none",
              padding: "14px 6px", borderBottom: "1px solid rgba(255,255,255,0.06)",
              cursor: "pointer",
            }}
          >
            {s.name}
          </button>
        ))}
        {results.length === 0 && (
          <div style={{ color: "#7f93a8", padding: "14px 6px" }}>No match.</div>
        )}
      </div>
    </div>
  );
}

// Q&A text floats directly on the map — no panel background, per the spec.
function QAOverlay({ q, index, total, targetName, onAnswer }) {
  return (
    <div
      style={{
        position: "absolute", left: 0, right: 0, bottom: 0, zIndex: 25,
        padding: "0 24px 40px", pointerEvents: "none",
      }}
    >
      <div style={{ pointerEvents: "auto", maxWidth: 460, margin: "0 auto" }}>
        <div style={{ fontSize: 12, color: "#7fd0ff", letterSpacing: 1,
                      textShadow: "0 1px 8px rgba(0,0,0,0.8)", marginBottom: 6 }}>
          {index + 1} / {total} · to {targetName}
        </div>
        <div style={{ fontSize: 26, fontWeight: 800, color: "#f2fbff", lineHeight: 1.15,
                      textShadow: "0 2px 14px rgba(0,0,0,0.85)" }}>
          {q.prompt}
        </div>
        <div style={{ fontSize: 13.5, color: "#b7d8ee", marginTop: 6,
                      textShadow: "0 1px 8px rgba(0,0,0,0.8)" }}>
          {q.hint}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 18 }}>
          {q.options.map((o) => (
            <button
              key={o.label}
              onClick={() => onAnswer(o.value)}
              style={{
                fontSize: 15, fontWeight: 700, color: "#eaf6ff",
                background: "rgba(255,255,255,0.10)",
                border: "1px solid rgba(159,216,255,0.35)",
                borderRadius: 999, padding: "10px 18px", cursor: "pointer",
                backdropFilter: "blur(2px)",
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function JourneyOverlay({
  leg, settled, currentName, targetName, remainingKm, stopsUsed, maxStops,
  hops, status, onStartVehicle, onTake, onReset, currentId,
}) {
  if (settled) {
    return (
      <BottomSheet>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 15, color: "#7fd0ff" }}>You&apos;ve arrived</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: "#f2fbff", margin: "4px 0 14px" }}>
            {targetName}
          </div>
          <button onClick={onReset} style={darkBtn}>new journey</button>
        </div>
      </BottomSheet>
    );
  }

  if (leg === "walk") {
    return (
      <BottomSheet>
        <div style={{ fontSize: 12, color: "#7fd0ff", letterSpacing: 1 }}>ON FOOT</div>
        <div style={{ fontSize: 20, fontWeight: 800, color: "#f2fbff", margin: "4px 0 2px" }}>
          Walk to {currentName}
        </div>
        <div style={{ fontSize: 13.5, color: "#b7d8ee", marginBottom: 14 }}>
          Head to the platform, then Silk shows your next moves toward {targetName}.
        </div>
        <button onClick={onStartVehicle} style={darkBtn}>I&apos;m at the station →</button>
      </BottomSheet>
    );
  }

  // vehicle leg
  const overStops = maxStops != null && stopsUsed >= maxStops;
  return (
    <BottomSheet>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <div style={{ fontSize: 12, color: "#7fd0ff", letterSpacing: 1 }}>
            AT {currentName?.toUpperCase()}
          </div>
          <div style={{ fontSize: 13.5, color: "#b7d8ee" }}>
            {Math.round(remainingKm)} km to {targetName} · {stopsUsed} stop{stopsUsed === 1 ? "" : "s"}
          </div>
        </div>
      </div>

      {overStops ? (
        <div style={{ color: "#ffd0a0", fontSize: 13.5, marginTop: 12 }}>
          Stop limit reached ({maxStops}). <button onClick={onReset} style={linkBtn}>reset</button> to continue.
        </div>
      ) : hops.length === 0 ? (
        <div style={{ color: "#9fb6cc", fontSize: 13.5, marginTop: 14 }}>
          {status === "loading" ? "Finding your next moves…" : "No onward hops toward the target from here."}
        </div>
      ) : (
        <div style={{ marginTop: 12, maxHeight: 260, overflowY: "auto" }}>
          {hops.map((h, i) => (
            <HopRow key={i} hop={h} currentId={currentId} onTake={onTake} />
          ))}
        </div>
      )}
    </BottomSheet>
  );
}

function HopRow({ hop, currentId, onTake }) {
  const toneColor = hop.tone === "fast" ? "#5aa0ff" : hop.tone === "slow" ? "#9fb6cc" : "#e8a13c";
  return (
    <div
      style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "11px 4px", borderBottom: "1px solid rgba(255,255,255,0.07)",
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11.5, fontWeight: 800, color: toneColor }}>{hop.clsLabel}</span>
          <span style={{ fontSize: 15, fontWeight: 600, color: "#eaf6ff", whiteSpace: "nowrap",
                         overflow: "hidden", textOverflow: "ellipsis" }}>
            → {hop.arriveName}
          </span>
        </div>
        <div style={{ fontSize: 12, color: "#8fb0c8", marginTop: 2 }}>
          <span style={{ color: "#4ade80" }}>−{Math.round(hop.progressKm)} km</span>
          {hop.time ? ` · ${hop.time}` : ""}
          {hop.delayMin ? ` · +${hop.delayMin}′` : ""}
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
        <a
          href={buyLink(hop, currentId)}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          style={{
            fontSize: 12.5, fontWeight: 700, color: "#9fd8ff", textDecoration: "none",
            border: "1px solid rgba(159,216,255,0.35)", borderRadius: 9, padding: "5px 9px",
          }}
        >
          {hop.price != null ? `€${hop.price}` : "buy"}
        </a>
        <button onClick={() => onTake(hop)} style={{ ...darkBtn, padding: "6px 12px", fontSize: 12.5 }}>
          board
        </button>
      </div>
    </div>
  );
}

// A translucent bottom sheet that sits over the map.
function BottomSheet({ children }) {
  return (
    <div
      style={{
        position: "absolute", left: 0, right: 0, bottom: 0, zIndex: 25,
        background: "linear-gradient(to top, rgba(8,12,20,0.94), rgba(8,12,20,0.78))",
        backdropFilter: "blur(6px)",
        borderTop: "1px solid rgba(159,216,255,0.15)",
        borderRadius: "20px 20px 0 0",
        padding: "18px 20px calc(20px + env(safe-area-inset-bottom))",
      }}
    >
      <div style={{ maxWidth: 460, margin: "0 auto" }}>{children}</div>
    </div>
  );
}

// ---- shared inline styles ----
function pillStyle(bg) {
  return {
    display: "inline-flex", alignItems: "center",
    fontSize: 12.5, fontWeight: 700, color: "#eaf6ff",
    background: bg, border: "1px solid rgba(255,255,255,0.18)",
    borderRadius: 999, padding: "7px 12px", cursor: "pointer",
    backdropFilter: "blur(3px)", textShadow: "0 1px 4px rgba(0,0,0,0.5)",
  };
}
const darkBtn = {
  fontSize: 14.5, fontWeight: 700, color: "#0b0f17", background: "#9fd8ff",
  border: "none", borderRadius: 12, padding: "11px 16px", cursor: "pointer",
};
const linkBtn = {
  background: "none", border: "none", color: "#9fd8ff", cursor: "pointer",
  textDecoration: "underline", fontSize: 13.5, padding: 0,
};
