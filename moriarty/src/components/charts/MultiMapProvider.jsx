import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { X, MapPin, BarChart3, Image, Video, Pencil, Save, Play, Pause, SkipForward, Trash2, Route } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const MultiProviderMap = () => {
  const [provider, setProvider] = useState('mapbox');
  const [annotations, setAnnotations] = useState([]);
  const [paths, setPaths] = useState([]);
  const [drawingMode, setDrawingMode] = useState(null);
  const [drawingPoints, setDrawingPoints] = useState([]);
  const [mapCenter, setMapCenter] = useState({ lat: 20, lng: 0, zoom: 2 });
  const [playback, setPlayback] = useState({ isPlaying: false, currentStep: 0, steps: [] });
  const [draggedAnnotation, setDraggedAnnotation] = useState(null);
  
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const playbackTimerRef = useRef(null);

  // Sample chart data
  const lineData = [
    { name: 'Mon', value: 20 },
    { name: 'Tue', value: 35 },
    { name: 'Wed', value: 28 },
    { name: 'Thu', value: 42 },
    { name: 'Fri', value: 38 }
  ];

  const pieData = [
    { name: 'A', value: 400 },
    { name: 'B', value: 300 },
    { name: 'C', value: 200 }
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28'];

  // API Keys - Replace with your actual keys
  const API_KEYS = {
    cesium: 'YOUR_CESIUM_KEY',
    mapbox: 'YOUR_MAPBOX_KEY',
    tomtom: 'YOUR_TOMTOM_KEY',
    openweather: 'YOUR_OPENWEATHER_KEY'
  };

  useEffect(() => {
    initializeMap();
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current = null;
      }
    };
  }, [provider]);

  useEffect(() => {
    if (mapInstanceRef.current) {
      updateMapView();
      redrawPaths();
    }
  }, [mapCenter, paths]);

  useEffect(() => {
    const savedData = localStorage.getItem('mapAnnotations');
    if (savedData) {
      const parsed = JSON.parse(savedData);
      setAnnotations(parsed.annotations || []);
      setPaths(parsed.paths || []);
    }
  }, []);

  const saveToLocalStorage = useCallback(() => {
    localStorage.setItem('mapAnnotations', JSON.stringify({
      annotations,
      paths
    }));
  }, [annotations, paths]);

  const initializeMap = () => {
    if (!mapContainerRef.current) return;
    mapContainerRef.current.innerHTML = '';

    switch (provider) {
      case 'mapbox':
        initMapbox();
        break;
      case 'cesium':
        initCesium();
        break;
      case 'tomtom':
        initTomTom();
        break;
      case 'openweather':
        initOpenWeather();
        break;
    }
  };

  const initMapbox = () => {
    const mapDiv = document.createElement('div');
    mapDiv.style.width = '100%';
    mapDiv.style.height = '100%';
    mapContainerRef.current.appendChild(mapDiv);

    if (window.mapboxgl) {
      window.mapboxgl.accessToken = API_KEYS.mapbox;
      mapInstanceRef.current = new window.mapboxgl.Map({
        container: mapDiv,
        style: 'mapbox://styles/mapbox/streets-v12',
        center: [mapCenter.lng, mapCenter.lat],
        zoom: mapCenter.zoom
      });
      mapInstanceRef.current.on('load', redrawPaths);
    } else {
      const script = document.createElement('script');
      script.src = 'https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js';
      script.onload = () => {
        const link = document.createElement('link');
        link.href = 'https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css';
        link.rel = 'stylesheet';
        document.head.appendChild(link);
        initMapbox();
      };
      document.head.appendChild(script);
    }
  };

  const initCesium = () => {
    const cesiumDiv = document.createElement('div');
    cesiumDiv.style.width = '100%';
    cesiumDiv.style.height = '100%';
    mapContainerRef.current.appendChild(cesiumDiv);

    if (window.Cesium) {
      window.Cesium.Ion.defaultAccessToken = API_KEYS.cesium;
      mapInstanceRef.current = new window.Cesium.Viewer(cesiumDiv, {
        terrainProvider: window.Cesium.createWorldTerrain()
      });
      mapInstanceRef.current.camera.setView({
        destination: window.Cesium.Cartesian3.fromDegrees(mapCenter.lng, mapCenter.lat, 10000000)
      });
    } else {
      const script = document.createElement('script');
      script.src = 'https://cesium.com/downloads/cesiumjs/releases/1.109/Build/Cesium/Cesium.js';
      script.onload = () => {
        const link = document.createElement('link');
        link.href = 'https://cesium.com/downloads/cesiumjs/releases/1.109/Build/Cesium/Widgets/widgets.css';
        link.rel = 'stylesheet';
        document.head.appendChild(link);
        initCesium();
      };
      document.head.appendChild(script);
    }
  };

  const initTomTom = () => {
    const mapDiv = document.createElement('div');
    mapDiv.style.width = '100%';
    mapDiv.style.height = '100%';
    mapContainerRef.current.appendChild(mapDiv);

    if (window.tt) {
      mapInstanceRef.current = window.tt.map({
        key: API_KEYS.tomtom,
        container: mapDiv,
        center: [mapCenter.lng, mapCenter.lat],
        zoom: mapCenter.zoom
      });
      mapInstanceRef.current.on('load', redrawPaths);
    } else {
      const script = document.createElement('script');
      script.src = 'https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.25.0/maps/maps-web.min.js';
      script.onload = () => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.25.0/maps/maps.css';
        document.head.appendChild(link);
        initTomTom();
      };
      document.head.appendChild(script);
    }
  };

  const initOpenWeather = () => {
    const mapDiv = document.createElement('div');
    mapDiv.style.width = '100%';
    mapDiv.style.height = '100%';
    mapContainerRef.current.appendChild(mapDiv);

    if (window.L) {
      mapInstanceRef.current = window.L.map(mapDiv).setView([mapCenter.lat, mapCenter.lng], mapCenter.zoom);
      window.L.tileLayer(`https://tile.openweathermap.org/map/temp_new/{z}/{x}/{y}.png?appid=${API_KEYS.openweather}`, {
        maxZoom: 19,
        attribution: '© OpenWeather'
      }).addTo(mapInstanceRef.current);
      redrawPaths();
    } else {
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      script.onload = () => {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);
        initOpenWeather();
      };
      document.head.appendChild(script);
    }
  };

  const updateMapView = () => {
    if (!mapInstanceRef.current) return;

    if (provider === 'mapbox' && mapInstanceRef.current.flyTo) {
      mapInstanceRef.current.flyTo({
        center: [mapCenter.lng, mapCenter.lat],
        zoom: mapCenter.zoom,
        duration: 2000
      });
    } else if (provider === 'tomtom' && mapInstanceRef.current.flyTo) {
      mapInstanceRef.current.flyTo({
        center: [mapCenter.lng, mapCenter.lat],
        zoom: mapCenter.zoom
      });
    } else if (provider === 'openweather' && mapInstanceRef.current.setView) {
      mapInstanceRef.current.setView([mapCenter.lat, mapCenter.lng], mapCenter.zoom);
    } else if (provider === 'cesium' && window.Cesium && mapInstanceRef.current.camera) {
      mapInstanceRef.current.camera.flyTo({
        destination: window.Cesium.Cartesian3.fromDegrees(mapCenter.lng, mapCenter.lat, 10000000 / Math.pow(2, mapCenter.zoom - 2))
      });
    }
  };

  const redrawPaths = () => {
    if (!mapInstanceRef.current) return;

    paths.forEach((path, index) => {
      if (provider === 'mapbox' && mapInstanceRef.current.getSource) {
        const sourceId = `path-${index}`;
        if (mapInstanceRef.current.getSource(sourceId)) {
          mapInstanceRef.current.removeLayer(sourceId);
          mapInstanceRef.current.removeSource(sourceId);
        }

        mapInstanceRef.current.addSource(sourceId, {
          type: 'geojson',
          data: {
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: path.points.map(p => [p.lng, p.lat])
            }
          }
        });

        mapInstanceRef.current.addLayer({
          id: sourceId,
          type: 'line',
          source: sourceId,
          paint: {
            'line-color': path.color || '#3b82f6',
            'line-width': path.width || 3,
            'line-dasharray': path.dashed ? [2, 2] : [1, 0]
          }
        });
      } else if (provider === 'openweather' && window.L && mapInstanceRef.current.addLayer) {
        const latlngs = path.points.map(p => [p.lat, p.lng]);
        const polyline = window.L.polyline(latlngs, {
          color: path.color || '#3b82f6',
          weight: path.width || 3,
          dashArray: path.dashed ? '5, 5' : null
        }).addTo(mapInstanceRef.current);
      } else if (provider === 'tomtom' && window.tt && mapInstanceRef.current.addLayer) {
        // TomTom path drawing would go here
      }
    });
  };

  const latLngToScreen = (lat, lng) => {
    if (!mapContainerRef.current) return { x: 50, y: 50 };
    
    const rect = mapContainerRef.current.getBoundingClientRect();
    const centerLat = mapCenter.lat;
    const centerLng = mapCenter.lng;
    
    // Simple mercator projection approximation
    const scale = Math.pow(2, mapCenter.zoom) * 256 / 360;
    const x = ((lng - centerLng) * scale + rect.width / 2) / rect.width * 100;
    const y = ((centerLat - lat) * scale + rect.height / 2) / rect.height * 100;
    
    return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) };
  };

  const screenToLatLng = (x, y) => {
    if (!mapContainerRef.current) return { lat: 0, lng: 0 };
    
    const rect = mapContainerRef.current.getBoundingClientRect();
    const scale = Math.pow(2, mapCenter.zoom) * 256 / 360;
    
    const lng = mapCenter.lng + ((x * rect.width / 100) - rect.width / 2) / scale;
    const lat = mapCenter.lat - ((y * rect.height / 100) - rect.height / 2) / scale;
    
    return { lat, lng };
  };

  const addAnnotation = (type) => {
    const coords = screenToLatLng(50, 50);
    const newAnnotation = {
      id: Date.now(),
      type,
      lat: coords.lat,
      lng: coords.lng,
      content: type === 'image' ? 'https://via.placeholder.com/150' : 
               type === 'video' ? 'https://www.youtube.com/embed/dQw4w9WgXcQ' : '',
      chartType: type === 'chart' ? 'line' : null
    };
    setAnnotations([...annotations, newAnnotation]);
  };

  const removeAnnotation = (id) => {
    setAnnotations(annotations.filter(ann => ann.id !== id));
  };

  const handleMapClick = (e) => {
    if (!drawingMode) return;

    const rect = mapContainerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    const coords = screenToLatLng(x, y);

    setDrawingPoints([...drawingPoints, coords]);
  };

  const finishDrawing = () => {
    if (drawingPoints.length < 2) {
      setDrawingMode(null);
      setDrawingPoints([]);
      return;
    }

    const newPath = {
      id: Date.now(),
      type: drawingMode,
      points: drawingPoints,
      color: '#3b82f6',
      width: 3,
      dashed: drawingMode === 'dashed'
    };

    setPaths([...paths, newPath]);
    setDrawingMode(null);
    setDrawingPoints([]);
  };

  const removePath = (id) => {
    setPaths(paths.filter(p => p.id !== id));
  };

  const addPlaybackStep = () => {
    const newStep = {
      id: Date.now(),
      provider,
      center: { ...mapCenter },
      annotations: [...annotations],
      paths: [...paths],
      description: `Step ${playback.steps.length + 1}`
    };
    setPlayback({ ...playback, steps: [...playback.steps, newStep] });
  };

  const playStep = (stepIndex) => {
    if (stepIndex >= playback.steps.length) {
      setPlayback({ ...playback, isPlaying: false, currentStep: 0 });
      return;
    }

    const step = playback.steps[stepIndex];
    setProvider(step.provider);
    setMapCenter(step.center);
    setAnnotations(step.annotations);
    setPaths(step.paths);
    setPlayback({ ...playback, currentStep: stepIndex });
  };

  const togglePlayback = () => {
    if (playback.isPlaying) {
      clearInterval(playbackTimerRef.current);
      setPlayback({ ...playback, isPlaying: false });
    } else {
      setPlayback({ ...playback, isPlaying: true });
      playbackTimerRef.current = setInterval(() => {
        setPlayback(prev => {
          const nextStep = prev.currentStep + 1;
          if (nextStep >= prev.steps.length) {
            clearInterval(playbackTimerRef.current);
            return { ...prev, isPlaying: false, currentStep: 0 };
          }
          playStep(nextStep);
          return { ...prev, currentStep: nextStep };
        });
      }, 3000);
    }
  };

  const handleAnnotationDragStart = (e, annotation) => {
    e.stopPropagation();
    setDraggedAnnotation(annotation);
  };

  const handleAnnotationDrag = (e) => {
    if (!draggedAnnotation) return;
    
    const rect = mapContainerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    const coords = screenToLatLng(x, y);

    setAnnotations(annotations.map(ann => 
      ann.id === draggedAnnotation.id 
        ? { ...ann, lat: coords.lat, lng: coords.lng }
        : ann
    ));
  };

  const handleAnnotationDragEnd = () => {
    setDraggedAnnotation(null);
  };

  const renderChart = (annotation) => {
    switch (annotation.chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" style={{ fontSize: '12px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={60} fill="#8884d8" dataKey="value" label>
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );
      default:
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" style={{ fontSize: '12px' }} />
              <YAxis style={{ fontSize: '12px' }} />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        );
    }
  };

  return (
    <div className="w-full h-screen flex flex-col bg-gray-100">
      {/* Header Controls */}
      <div className="bg-white shadow-md p-3 flex items-center gap-3 z-10 flex-wrap">
        <h1 className="text-lg font-bold text-gray-800">Multi-Provider Map</h1>
        
        <Select value={provider} onValueChange={setProvider}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Provider" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="mapbox">Mapbox</SelectItem>
            <SelectItem value="cesium">Cesium</SelectItem>
            <SelectItem value="tomtom">TomTom</SelectItem>
            <SelectItem value="openweather">OpenWeather</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex gap-2">
          <button
            onClick={() => addAnnotation('marker')}
            className="flex items-center gap-1 px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition text-sm"
          >
            <MapPin className="w-4 h-4" />
            Marker
          </button>
          <button
            onClick={() => addAnnotation('chart')}
            className="flex items-center gap-1 px-3 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition text-sm"
          >
            <BarChart3 className="w-4 h-4" />
            Chart
          </button>
          <button
            onClick={() => addAnnotation('image')}
            className="flex items-center gap-1 px-3 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 transition text-sm"
          >
            <Image className="w-4 h-4" />
            Image
          </button>
          <button
            onClick={() => addAnnotation('video')}
            className="flex items-center gap-1 px-3 py-2 bg-pink-500 text-white rounded hover:bg-pink-600 transition text-sm"
          >
            <Video className="w-4 h-4" />
            Video
          </button>
        </div>

        <div className="flex gap-2 border-l pl-3">
          <button
            onClick={() => setDrawingMode('path')}
            className={`flex items-center gap-1 px-3 py-2 rounded transition text-sm ${
              drawingMode === 'path' ? 'bg-orange-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            <Route className="w-4 h-4" />
            Path
          </button>
          <button
            onClick={() => setDrawingMode('dashed')}
            className={`flex items-center gap-1 px-3 py-2 rounded transition text-sm ${
              drawingMode === 'dashed' ? 'bg-orange-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            <Pencil className="w-4 h-4" />
            Dashed
          </button>
          {drawingMode && (
            <button
              onClick={finishDrawing}
              className="flex items-center gap-1 px-3 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition text-sm"
            >
              Finish
            </button>
          )}
        </div>

        <div className="flex gap-2 border-l pl-3 ml-auto">
          <button
            onClick={addPlaybackStep}
            className="flex items-center gap-1 px-3 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600 transition text-sm"
          >
            <Save className="w-4 h-4" />
            Save Step
          </button>
          <button
            onClick={togglePlayback}
            disabled={playback.steps.length === 0}
            className="flex items-center gap-1 px-3 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600 transition text-sm disabled:opacity-50"
          >
            {playback.isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {playback.isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={saveToLocalStorage}
            className="flex items-center gap-1 px-3 py-2 bg-gray-700 text-white rounded hover:bg-gray-800 transition text-sm"
          >
            <Save className="w-4 h-4" />
            Save All
          </button>
        </div>
      </div>

      {/* Map Container with Annotations */}
      <div 
        className="flex-1 relative"
        onClick={handleMapClick}
        onMouseMove={handleAnnotationDrag}
        onMouseUp={handleAnnotationDragEnd}
      >
        {/* Map Layer */}
        <div ref={mapContainerRef} className="absolute inset-0 bg-gray-200" />

        {/* Path Preview */}
        {drawingMode && drawingPoints.length > 0 && (
          <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: 5 }}>
            <polyline
              points={drawingPoints.map(p => {
                const screen = latLngToScreen(p.lat, p.lng);
                return `${screen.x}%,${screen.y}%`;
              }).join(' ')}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="3"
              strokeDasharray={drawingMode === 'dashed' ? '5,5' : '0'}
            />
            {drawingPoints.map((p, i) => {
              const screen = latLngToScreen(p.lat, p.lng);
              return (
                <circle
                  key={i}
                  cx={`${screen.x}%`}
                  cy={`${screen.y}%`}
                  r="5"
                  fill="#f59e0b"
                />
              );
            })}
          </svg>
        )}

        {/* Annotation Layer */}
        <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 10 }}>
          {annotations.map((annotation) => {
            const pos = latLngToScreen(annotation.lat, annotation.lng);
            return (
              <div
                key={annotation.id}
                className="absolute pointer-events-auto cursor-move"
                style={{
                  left: `${pos.x}%`,
                  top: `${pos.y}%`,
                  transform: 'translate(-50%, -50%)'
                }}
                onMouseDown={(e) => handleAnnotationDragStart(e, annotation)}
              >
                {annotation.type === 'marker' ? (
                  <Card className="p-3 bg-white shadow-lg min-w-48">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-5 h-5 text-red-500" />
                        <div>
                          <h3 className="font-semibold text-sm">Location</h3>
                          <p className="text-xs text-gray-600">
                            {annotation.lat.toFixed(4)}, {annotation.lng.toFixed(4)}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeAnnotation(annotation.id)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </Card>
                ) : annotation.type === 'chart' ? (
                  <Card className="p-4 bg-white shadow-lg">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex gap-2">
                        <h3 className="font-semibold text-sm">Chart</h3>
                        <select
                          value={annotation.chartType}
                          onChange={(e) => {
                            setAnnotations(annotations.map(a => 
                              a.id === annotation.id ? { ...a, chartType: e.target.value } : a
                            ));
                          }}
                          className="text-xs border rounded px-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <option value="line">Line</option>
                          <option value="bar">Bar</option>
                          <option value="pie">Pie</option>
                        </select>
                      </div>
                      <button
                        onClick={() => removeAnnotation(annotation.id)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="w-64 h-40">
                      {renderChart(annotation)}
                    </div>
                  </Card>
                ) : annotation.type === 'image' ? (
                  <Card className="p-2 bg-white shadow-lg">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="font-semibold text-sm">Image</h3>
                      <button
                        onClick={() => removeAnnotation(annotation.id)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <img src={annotation.content} alt="Annotation" className="w-48 h-32 object-cover rounded" />
                  </Card>
                ) : annotation.type === 'video' ? (
                  <Card className="p-2 bg-white shadow-lg">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="font-semibold text-sm">Video</h3>
                      <button
                        onClick={() => removeAnnotation(annotation.id)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <iframe
                      src={annotation.content}
                      className="w-64 h-40 rounded"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                    />
                  </Card>
                ) : null}
              </div>
            );
          })}
        </div>

        {/* Info Panel */}
        <Card className="absolute bottom-4 left-4 p-3 bg-white shadow-lg max-w-xs">
          <h3 className="font-semibold text-sm mb-2">Status</h3>
          <div className="text-xs text-gray-600 space-y-1">
            <p>Provider: <span className="font-medium">{provider}</span></p>
            <p>Annotations: <span className="font-medium">{annotations.length}</span></p>
            <p>Paths: <span className="font-medium">{paths.length}</span></p>
            <p>Steps: <span className="font-medium">{playback.steps.length}</span></p>
            {playback.isPlaying && (
              <p className="text-indigo-600 font-medium">
                Playing Step {playback.currentStep + 1}/{playback.steps.length}
              </p>
            )}
            {drawingMode && (
              <p className="text-orange-600 font-medium">
                Drawing {drawingMode} ({drawingPoints.length} points)
              </p>
            )}
          </div>
        </Card>

        {/* Playback Controls */}
        {playback.steps.length > 0 && (
          <Card className="absolute bottom-4 right-4 p-3 bg-white shadow-lg">
            <h3 className="font-semibold text-sm mb-2">Playback Steps</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {playback.steps.map((step, index) => (
                <div key={step.id} className="flex items-center gap-2 text-xs">
                  <button
                    onClick={() => playStep(index)}
                    className={`flex-1 px-2 py-1 rounded text-left ${
                      playback.currentStep === index ? 'bg-indigo-500 text-white' : 'bg-gray-100 hover:bg-gray-200'
                    }`}
                  >
                    {index + 1}. {step.provider} - {step.description}
                  </button>
                  <button
                    onClick={() => setPlayback({ ...playback, steps: playback.steps.filter((_, i) => i !== index) })}
                    className="text-red-500 hover:text-red-700"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Path List */}
        {paths.length > 0 && (
          <Card className="absolute top-4 right-4 p-3 bg-white shadow-lg max-w-xs">
            <h3 className="font-semibold text-sm mb-2">Paths & Routes</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {paths.map((path, index) => (
                <div key={path.id} className="flex items-center justify-between text-xs bg-gray-50 p-2 rounded">
                  <div>
                    <span className="font-medium">Path {index + 1}</span>
                    <span className="text-gray-500 ml-2">({path.points.length} points)</span>
                  </div>
                  <button
                    onClick={() => removePath(path.id)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default MultiProviderMap;