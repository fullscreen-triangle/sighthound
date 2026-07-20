import React, { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { Map as MapGL } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

// Silk's map surface: a Mapbox basemap with a glowing deck.gl position dot,
// starting at street level (zoom ~18). deck.gl renders the glow / hops / trail
// as overlay layers on top of the Mapbox tiles.
//
// Props:
//   position   [lat, lon]                 the glowing dot (user / current station)
//   target     [lat, lon] | null          destination marker (coral)
//   hops       [{coord:[lat,lon], tone}]  candidate next-hop endpoints
//   trail      [[lat,lon], ...]           travelled path
//   pitch, zoom                           view tuning (zoom defaults to 18)
//   interactive                           allow pan/zoom
export default function SilkGlobe({
  position,
  target = null,
  hops = [],
  trail = [],
  pitch = 55,
  zoom = 18,
  interactive = true,
  onReady = null,
}) {
  const [t, setT] = useState(0);
  const raf = useRef(null);

  // pulse clock for the glowing dot
  useEffect(() => {
    let start;
    const tick = (ts) => {
      if (!start) start = ts;
      setT((ts - start) / 1000);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, []);

  const initialViewState = useMemo(
    () => ({
      longitude: position ? position[1] : 11.5,
      latitude: position ? position[0] : 50.5,
      zoom,
      pitch,
      bearing: 0,
    }),
    // recenter only when the target position identity changes
    [position && position[0], position && position[1], zoom, pitch]
  );

  const pulse = 0.5 + 0.5 * Math.sin(t * 2.2); // 0..1

  const layers = useMemo(() => {
    const L = [];

    // travelled trail (soft cyan)
    if (trail.length > 1) {
      L.push(
        new PathLayer({
          id: "silk-trail",
          data: [{ path: trail.map(([lat, lon]) => [lon, lat]) }],
          getPath: (d) => d.path,
          getColor: [90, 200, 250, 200],
          getWidth: 4,
          widthUnits: "pixels",
          widthMinPixels: 2,
        })
      );
    }

    // candidate hop rays
    if (position && hops.length) {
      L.push(
        new PathLayer({
          id: "silk-hops",
          data: hops
            .filter((h) => h.coord)
            .map((h) => ({
              path: [
                [position[1], position[0]],
                [h.coord[1], h.coord[0]],
              ],
              tone: h.tone,
            })),
          getPath: (d) => d.path,
          getColor: (d) =>
            d.tone === "fast" ? [80, 150, 255, 170] :
            d.tone === "slow" ? [150, 170, 190, 150] :
            [230, 160, 60, 160],
          getWidth: 3,
          widthUnits: "pixels",
          widthMinPixels: 1.5,
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-hop-dots",
          data: hops.filter((h) => h.coord),
          getPosition: (h) => [h.coord[1], h.coord[0]],
          getRadius: 6,
          radiusUnits: "pixels",
          getFillColor: (h) =>
            h.tone === "fast" ? [80, 150, 255, 235] :
            h.tone === "slow" ? [150, 170, 190, 225] :
            [230, 160, 60, 235],
          stroked: true,
          getLineColor: [255, 255, 255, 210],
          lineWidthUnits: "pixels",
          getLineWidth: 1.5,
        })
      );
    }

    // destination marker (coral halo)
    if (target) {
      L.push(
        new ScatterplotLayer({
          id: "silk-target",
          data: [{ p: [target[1], target[0]] }],
          getPosition: (d) => d.p,
          getRadius: 10,
          radiusUnits: "pixels",
          getFillColor: [232, 113, 10, 245],
          stroked: true,
          getLineColor: [255, 255, 255, 235],
          lineWidthUnits: "pixels",
          getLineWidth: 2,
        })
      );
    }

    // glowing position dot — pulsing layered halos
    if (position) {
      const p = [position[1], position[0]];
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-outer",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 30 + 12 * pulse,
          radiusUnits: "pixels",
          getFillColor: [90, 200, 250, Math.round(45 + 35 * pulse)],
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-mid",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 16 + 5 * pulse,
          radiusUnits: "pixels",
          getFillColor: [120, 220, 255, 150],
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-core",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 7,
          radiusUnits: "pixels",
          getFillColor: [230, 250, 255, 255],
          stroked: true,
          getLineColor: [120, 220, 255, 230],
          lineWidthUnits: "pixels",
          getLineWidth: 2,
        })
      );
    }

    return L;
  }, [position, target, hops, trail, pulse]);

  return (
    <DeckGL
      initialViewState={initialViewState}
      controller={interactive}
      layers={layers}
      onLoad={onReady || undefined}
      style={{ position: "absolute", inset: 0 }}
    >
      <MapGL
        reuseMaps
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle="mapbox://styles/mapbox/dark-v11"
      />
    </DeckGL>
  );
}
