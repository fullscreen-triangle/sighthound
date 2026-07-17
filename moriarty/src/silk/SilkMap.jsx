import React, { useEffect, useRef } from "react";
import { STATIONS } from "./network";

// Silk map: shows the user's current station, the destination, the journey
// travelled so far, and the candidate next hops. Reuses the project's existing
// pattern of loading Leaflet dynamically from CDN (see MapVisualization.jsx),
// so no new dependency is added and SSR is not an issue.
export default function SilkMap({ currentId, targetId, hops = [], trail = [] }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const ensureLeaflet = (cb) => {
      if (window.L) return cb();
      if (!document.getElementById("leaflet-css")) {
        const link = document.createElement("link");
        link.id = "leaflet-css";
        link.rel = "stylesheet";
        link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
        document.head.appendChild(link);
      }
      let script = document.getElementById("leaflet-js");
      if (!script) {
        script = document.createElement("script");
        script.id = "leaflet-js";
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
        script.onload = cb;
        document.head.appendChild(script);
      } else {
        script.addEventListener("load", cb, { once: true });
      }
    };

    const draw = () => {
      const L = window.L;
      if (!mapRef.current) {
        mapRef.current = L.map(containerRef.current, {
          zoomControl: false,
          attributionControl: false,
        });
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
          { maxZoom: 19 }
        ).addTo(mapRef.current);
      }
      const map = mapRef.current;

      // clear previous overlay layer
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
      }
      const group = L.layerGroup().addTo(map);
      layerRef.current = group;

      const cur = STATIONS[currentId];
      const tgt = STATIONS[targetId];
      const pts = [];

      // travelled trail (grey line through visited stations)
      if (trail.length > 1) {
        const line = trail.map((id) => STATIONS[id].coord);
        L.polyline(line, { color: "#9aa5b1", weight: 3, opacity: 0.8 }).addTo(group);
        line.forEach((c) => pts.push(c));
      }

      // candidate hops from current station (blue dashed rays)
      if (cur) {
        hops.forEach((h) => {
          const dest = STATIONS[h.arriveId];
          if (!dest) return;
          L.polyline([cur.coord, dest.coord], {
            color: "#1f6feb",
            weight: 2,
            opacity: 0.6,
            dashArray: "5,6",
          }).addTo(group);
          L.circleMarker(dest.coord, {
            radius: 5, fillColor: "#1f6feb", color: "#fff", weight: 2, fillOpacity: 0.9,
          }).bindTooltip(`${h.clsLabel} → ${h.arriveName}`, { direction: "top" }).addTo(group);
          pts.push(dest.coord);
        });
      }

      // destination star (coral)
      if (tgt) {
        L.circleMarker(tgt.coord, {
          radius: 8, fillColor: "#e8710a", color: "#fff", weight: 2, fillOpacity: 1,
        }).bindTooltip(`★ ${tgt.name}`, { permanent: false, direction: "top" }).addTo(group);
        pts.push(tgt.coord);
      }

      // current position (dark, large)
      if (cur) {
        L.circleMarker(cur.coord, {
          radius: 9, fillColor: "#111827", color: "#fff", weight: 2, fillOpacity: 1,
        }).bindTooltip(`You: ${cur.name}`, { permanent: false, direction: "top" }).addTo(group);
        pts.push(cur.coord);
      }

      // fit to everything visible
      if (pts.length === 1) {
        map.setView(pts[0], 8);
      } else if (pts.length > 1) {
        map.fitBounds(pts, { padding: [40, 40] });
      }
    };

    ensureLeaflet(draw);
  }, [currentId, targetId, hops, trail]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", minHeight: "220px", background: "#eef1f5" }}
    />
  );
}
