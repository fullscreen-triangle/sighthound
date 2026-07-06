import React, { useEffect, useRef, useState, useMemo } from "react";
import MapGL from "react-map-gl";
import DeckGL from "@deck.gl/react";
import "mapbox-gl/dist/mapbox-gl.css";
import { createIsochroneLayer } from "@/layers/IsochroneLayer";
import { createHeatmapLayer } from "@/layers/HeatmapLayer";
import { createWindVectorLayer } from "@/layers/WeatherLayer";
import { createFWDCRoutingLayers } from "@/layers/RoutingLayer";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

export default function SandboxMap({
  initialViewState = {
    longitude: 11.5656,
    latitude: 48.1351,
    zoom: 11,
    pitch: 0,
    bearing: 0,
  },
  layers = [],
  onMapLoad = null,
  onViewStateChange = null,
}) {
  const mapRef = useRef(null);
  const [viewState, setViewState] = useState(initialViewState);
  const [mapLoaded, setMapLoaded] = useState(false);

  const handleViewStateChange = (newViewState) => {
    setViewState(newViewState.viewState);
    if (onViewStateChange) {
      onViewStateChange(newViewState.viewState);
    }
  };

  const handleMapLoad = () => {
    setMapLoaded(true);
    if (onMapLoad) {
      onMapLoad(mapRef.current);
    }
  };

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <DeckGL
        initialViewState={viewState}
        controller={true}
        layers={layers}
        onViewStateChange={handleViewStateChange}
      >
        <MapGL
          ref={mapRef}
          {...viewState}
          onLoad={handleMapLoad}
          mapStyle="mapbox://styles/mapbox/streets-v12"
          mapboxAccessToken={MAPBOX_TOKEN}
          onMove={handleViewStateChange}
        />
      </DeckGL>
    </div>
  );
}
