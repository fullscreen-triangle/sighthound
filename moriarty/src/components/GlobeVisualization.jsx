import React, { useState, useEffect, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Globe = dynamic(() => import('react-globe.gl').then(m => m.default), {
  ssr: false,
  loading: () => <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#000' }}>Loading...</div>,
});

// Stable particle color accessor (module-level constant — not a hook, so it is
// safe to pass in JSX after conditional early returns).
const particlesColorAccessor = () => 'palegreen';

const satellite = typeof window !== 'undefined' ? require('satellite.js') : null;

const EARTH_RADIUS_KM = 6371;
const TIME_STEP = 3 * 1000;

export default function GlobeVisualization() {
  const globeEl = useRef();
  const [satData, setSatData] = useState();
  const [time, setTime] = useState(new Date());
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const frameTicker = () => {
      requestAnimationFrame(frameTicker);
      setTime(time => new Date(+time + TIME_STEP));
    };
    frameTicker();
  }, [mounted]);

  useEffect(() => {
    if (!mounted || !satellite) return;

    fetch('https://cdn.jsdelivr.net/npm/globe.gl/example/datasets/space-track-leo.txt')
      .then(r => r.text())
      .then(rawData => {
        const tleData = rawData
          .replace(/\r/g, '')
          .split(/\n(?=[^12])/)
          .filter(d => d)
          .map(tle => tle.split('\n'));

        const satData = tleData
          .map(([name, ...tle]) => ({
            satrec: satellite.twoline2satrec(...tle),
            name: name.trim().replace(/^0 /, ''),
          }))
          .filter(d => {
            try {
              return !!satellite.propagate(d.satrec, new Date())?.position;
            } catch {
              return false;
            }
          });

        setSatData(satData);
      })
      .catch(err => console.error('Error loading satellite data:', err));
  }, [mounted]);

  const particlesData = useMemo(() => {
    if (!satData || !satellite) return [];

    try {
      const gmst = satellite.gstime(time);
      const particles = satData
        .map(d => {
          try {
            const eci = satellite.propagate(d.satrec, time);
            if (eci?.position) {
              const gdPos = satellite.eciToGeodetic(eci.position, gmst);
              const lat = satellite.radiansToDegrees(gdPos.latitude);
              const lng = satellite.radiansToDegrees(gdPos.longitude);
              const alt = gdPos.height / EARTH_RADIUS_KM;
              return { ...d, lat, lng, alt };
            }
          } catch {
            return null;
          }
          return null;
        })
        .filter(d => d && !isNaN(d.lat) && !isNaN(d.lng) && !isNaN(d.alt));

      return [particles];
    } catch (err) {
      console.error('Error computing particles data:', err);
      return [];
    }
  }, [satData, time]);

  useEffect(() => {
    if (globeEl.current) {
      try {
        globeEl.current.pointOfView({ altitude: 3.5 });
      } catch {
        // Ignore errors during initialization
      }
    }
  }, []);

  if (!mounted) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#000' }}>Loading...</div>;
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', margin: 0 }}>
      <style>{`
        body { margin: 0; padding: 0; }
        #globe-container { width: 100%; height: 100%; }
      `}</style>

      <div id="globe-container">
        <Globe
          ref={globeEl}
          globeImageUrl="https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg"
          particlesData={particlesData}
          particleLabel="name"
          particleLat="lat"
          particleLng="lng"
          particleAltitude="alt"
          particlesColor={particlesColorAccessor}
          onGlobeReady={() => {
            if (globeEl.current) {
              try {
                globeEl.current.pointOfView({ altitude: 3.5 });
              } catch {
                // Ignore
              }
            }
          }}
        />
      </div>

      <div
        style={{
          position: 'absolute',
          fontSize: '12px',
          fontFamily: 'sans-serif',
          padding: '5px',
          borderRadius: '3px',
          backgroundColor: 'rgba(200, 200, 200, 0.1)',
          color: 'lavender',
          bottom: '10px',
          right: '10px',
          zIndex: 10,
        }}
      >
        {time.toString()}
      </div>
    </div>
  );
}
