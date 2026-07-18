import React, { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer, ScatterplotLayer, PathLayer } from "@deck.gl/layers";

// Silk's map surface: a dark, pitched deck.gl map with a glowing position dot.
// No Mapbox token required — the basemap is CARTO's free dark raster tiles,
// drawn through a deck.gl TileLayer, and everything else is deck.gl layers.
//
// Props:
//   position   [lat, lon]           the glowing dot (user / current station)
//   target     [lat, lon] | null    destination marker (coral)
//   hops       [{coord:[lat,lon], tone}] candidate next-hop endpoints
//   trail      [[lat,lon], ...]      travelled path
//   pitch, zoom                      view tuning
//   interactive                      allow pan/zoom
export default function SilkGlobe({
  position,
  target = null,
  hops = [],
  trail = [],
  pitch = 55,
  zoom = 5.4,
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

  const viewState = useMemo(
    () => ({
      longitude: position ? position[1] : 11.5,
      latitude: position ? position[0] : 50.5,
      zoom,
      pitch,
      bearing: 0,
    }),
    // recenter only when position identity changes, not every frame
    [position && position[0], position && position[1], zoom, pitch]
  );

  const pulse = 0.5 + 0.5 * Math.sin(t * 2.2); // 0..1

  const layers = useMemo(() => {
    const L = [];

    // 1. dark basemap (no token) — CARTO dark_all raster tiles
    L.push(
      new TileLayer({
        id: "silk-basemap",
        data: "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        minZoom: 0,
        maxZoom: 19,
        tileSize: 256,
        renderSubLayers: (props) => {
          const { boundingBox } = props.tile;
          return new BitmapLayer(props, {
            data: null,
            image: props.data,
            bounds: [
              boundingBox[0][0], boundingBox[0][1],
              boundingBox[1][0], boundingBox[1][1],
            ],
          });
        },
      })
    );

    // 2. travelled trail (soft cyan)
    if (trail.length > 1) {
      L.push(
        new PathLayer({
          id: "silk-trail",
          data: [{ path: trail.map(([lat, lon]) => [lon, lat]) }],
          getPath: (d) => d.path,
          getColor: [90, 200, 250, 180],
          getWidth: 3,
          widthUnits: "pixels",
          widthMinPixels: 2,
        })
      );
    }

    // 3. candidate hop rays (from position to each hop endpoint)
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
            d.tone === "fast" ? [80, 150, 255, 150] :
            d.tone === "slow" ? [130, 150, 170, 120] :
            [230, 160, 60, 140],
          getWidth: 2,
          widthUnits: "pixels",
          widthMinPixels: 1.5,
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-hop-dots",
          data: hops.filter((h) => h.coord),
          getPosition: (h) => [h.coord[1], h.coord[0]],
          getRadius: 5,
          radiusUnits: "pixels",
          getFillColor: (h) =>
            h.tone === "fast" ? [80, 150, 255, 230] :
            h.tone === "slow" ? [150, 170, 190, 220] :
            [230, 160, 60, 230],
          stroked: true,
          getLineColor: [255, 255, 255, 200],
          lineWidthUnits: "pixels",
          getLineWidth: 1,
        })
      );
    }

    // 4. destination marker (coral halo)
    if (target) {
      L.push(
        new ScatterplotLayer({
          id: "silk-target",
          data: [{ p: [target[1], target[0]] }],
          getPosition: (d) => d.p,
          getRadius: 9,
          radiusUnits: "pixels",
          getFillColor: [232, 113, 10, 240],
          stroked: true,
          getLineColor: [255, 255, 255, 230],
          lineWidthUnits: "pixels",
          getLineWidth: 2,
        })
      );
    }

    // 5. glowing position dot — layered halos that pulse
    if (position) {
      const p = [position[1], position[0]];
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-outer",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 26 + 10 * pulse,
          radiusUnits: "pixels",
          getFillColor: [90, 200, 250, Math.round(40 + 30 * pulse)],
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-mid",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 14 + 4 * pulse,
          radiusUnits: "pixels",
          getFillColor: [120, 220, 255, 130],
        })
      );
      L.push(
        new ScatterplotLayer({
          id: "silk-glow-core",
          data: [{ p }],
          getPosition: (d) => d.p,
          getRadius: 6,
          radiusUnits: "pixels",
          getFillColor: [220, 250, 255, 255],
          stroked: true,
          getLineColor: [120, 220, 255, 220],
          lineWidthUnits: "pixels",
          getLineWidth: 2,
        })
      );
    }

    return L;
  }, [position, target, hops, trail, pulse]);

  return (
    <div style={{ position: "absolute", inset: 0, background: "#0b0f17" }}>
      <DeckGL
        initialViewState={viewState}
        viewState={interactive ? undefined : viewState}
        controller={interactive}
        layers={layers}
        onLoad={onReady || undefined}
        style={{ background: "#0b0f17" }}
      />
    </div>
  );
}
