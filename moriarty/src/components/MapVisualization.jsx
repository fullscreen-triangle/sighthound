import React, { useEffect, useRef } from "react";

export default function MapVisualization({ features, results }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !features || features.length === 0) return;

    // Dynamically load Leaflet
    const loadLeaflet = async () => {
      if (!window.L) {
        const link = document.createElement("link");
        link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
        document.head.appendChild(link);

        const script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
        script.onload = initMap;
        document.head.appendChild(script);
      } else {
        initMap();
      }
    };

    const initMap = () => {
      const L = window.L;

      // Calculate bounds from features
      let minLat = 90,
        maxLat = -90,
        minLon = 180,
        maxLon = -180;

      features.forEach((f) => {
        const [lon, lat] = f.geometry.coordinates;
        minLat = Math.min(minLat, lat);
        maxLat = Math.max(maxLat, lat);
        minLon = Math.min(minLon, lon);
        maxLon = Math.max(maxLon, lon);
      });

      // Create map
      const map = L.map(containerRef.current).fitBounds([
        [minLat, minLon],
        [maxLat, maxLon],
      ]);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap",
        maxZoom: 19,
      }).addTo(map);

      // Draw trajectory
      const coords = features.map((f) => {
        const [lon, lat] = f.geometry.coordinates;
        return [lat, lon];
      });

      L.polyline(coords, {
        color: "#007acc",
        weight: 2,
        opacity: 0.8,
      }).addTo(map);

      // Add markers
      features.forEach((f, i) => {
        const [lon, lat] = f.geometry.coordinates;
        const popup = `
          <div style="font-size: 12px;">
            <b>Point ${i + 1}</b><br/>
            Lat: ${lat.toFixed(5)}°<br/>
            Lon: ${lon.toFixed(5)}°<br/>
            Alt: ${f.properties.elevation || "—"} m
          </div>
        `;

        L.circleMarker([lat, lon], {
          radius: 6,
          fillColor: "#06a77d",
          color: "#fff",
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8,
        })
          .bindPopup(popup)
          .addTo(map);
      });

      // If results exist, show error circles
      if (results && results.length > 0) {
        results.forEach((r, i) => {
          if (r.error_horizontal) {
            const [lon, lat] = features[i].geometry.coordinates;
            // Draw error circle
            L.circle([lat, lon], {
              radius: r.error_horizontal * 1000, // Convert to meters
              color: "#f48771",
              weight: 1,
              opacity: 0.3,
              fillOpacity: 0.1,
            }).addTo(map);

            // Draw estimated position
            if (r.estimated_lat && r.estimated_lon) {
              L.circleMarker([r.estimated_lat, r.estimated_lon], {
                radius: 5,
                fillColor: "#dcdcaa",
                color: "#fff",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8,
              })
                .bindPopup(
                  `Estimated<br/>Error: ${(r.error_horizontal * 100).toFixed(1)}cm`
                )
                .addTo(map);
            }
          }
        });
      }

      mapRef.current = map;
    };

    loadLeaflet();
  }, [features, results]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{
        background: "#f0f0f0",
        minHeight: "400px",
      }}
    />
  );
}
