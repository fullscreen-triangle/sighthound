import React, { useState, useEffect, useRef } from 'react';

export default function CompositionExplosion() {
  const [n, setN] = useState(1);
  const [d, setD] = useState(1);
  const lineChartRef = useRef(null);
  const heatmapRef = useRef(null);
  const barChartRef = useRef(null);

  const gamma = (n, d) => d * Math.pow(1 + d, n - 1);

  useEffect(() => {
    drawCharts();
  }, [n, d]);

  const drawLineChart = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const colors = ['#2d5016', '#4a7c1f', '#c85a54', '#d97060', '#8b4513', '#6b8e23', '#4682b4', '#8b008b'];
    const padding = 40;
    const graphWidth = canvas.width - 2 * padding;
    const graphHeight = canvas.height - 2 * padding;

    let maxGamma = 0;
    for (let ch = 1; ch <= d; ch++) {
      for (let depth = 1; depth <= 9; depth++) {
        maxGamma = Math.max(maxGamma, gamma(depth, ch));
      }
    }

    // Draw axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.stroke();

    // Draw grid
    ctx.strokeStyle = '#eee';
    for (let i = 0; i <= 9; i++) {
      const x = padding + (i / 9) * graphWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvas.height - padding);
      ctx.stroke();
    }

    // Draw lines
    for (let ch = 1; ch <= d; ch++) {
      ctx.strokeStyle = colors[ch - 1];
      ctx.lineWidth = ch === d ? 3 : 2;
      ctx.beginPath();

      for (let depth = 1; depth <= 9; depth++) {
        const g = gamma(depth, ch);
        const y = Math.log10(Math.max(g, 1));
        const x = padding + ((depth - 1) / 8) * graphWidth;
        const screenY = canvas.height - padding - (y / Math.log10(maxGamma + 1)) * graphHeight;

        if (depth === 1) {
          ctx.moveTo(x, screenY);
        } else {
          ctx.lineTo(x, screenY);
        }
      }
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia';
    ctx.textAlign = 'center';
    for (let i = 1; i <= 9; i++) {
      const x = padding + ((i - 1) / 8) * graphWidth;
      ctx.fillText(i, x, canvas.height - 20);
    }
    ctx.textAlign = 'center';
    ctx.fillText('Composition Depth (n)', canvas.width / 2, canvas.height - 5);
  };

  const drawHeatmap = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellSize = 40;
    const padding = 20;
    const cols = 9;
    const rows = d;

    let maxGamma = 0;
    for (let ch = 1; ch <= d; ch++) {
      for (let depth = 1; depth <= 9; depth++) {
        maxGamma = Math.max(maxGamma, gamma(depth, ch));
      }
    }

    const colormap = (val) => {
      const h = (1 - val) * 240;
      return `hsl(${h}, 100%, 50%)`;
    };

    for (let ch = 1; ch <= d; ch++) {
      for (let depth = 1; depth <= 9; depth++) {
        const g = gamma(depth, ch);
        const normalized = g / maxGamma;

        const x = padding + (depth - 1) * cellSize;
        const y = padding + (ch - 1) * cellSize;

        ctx.fillStyle = colormap(normalized);
        ctx.fillRect(x, y, cellSize - 2, cellSize - 2);

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize - 2, cellSize - 2);

        ctx.fillStyle = '#333';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const text = Math.floor(g).toString().length > 3 ? Math.floor(g / 1000).toFixed(0) + 'k' : Math.floor(g);
        ctx.fillText(text, x + cellSize / 2 - 1, y + cellSize / 2);
      }
    }
  };

  const drawBarChart = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const padding = 40;
    const barWidth = (canvas.width - 2 * padding) / 9;
    const maxHeight = canvas.height - 2 * padding;

    let maxGamma = 0;
    for (let depth = 1; depth <= 9; depth++) {
      maxGamma = Math.max(maxGamma, gamma(depth, d));
    }

    for (let depth = 1; depth <= 9; depth++) {
      const g = gamma(depth, d);
      const normalized = g / maxGamma;
      const height = normalized * maxHeight;

      const x = padding + (depth - 1) * barWidth;
      const y = canvas.height - padding - height;

      const color = depth <= n ? '#c85a54' : '#ddd';
      ctx.fillStyle = color;
      ctx.fillRect(x + 4, y, barWidth - 8, height);

      ctx.fillStyle = '#333';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const text = Math.floor(g).toLocaleString();
      ctx.fillText(text, x + barWidth / 2, y - 15);
    }

    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.stroke();
  };

  const drawCharts = () => {
    drawLineChart(lineChartRef.current);
    drawHeatmap(heatmapRef.current);
    drawBarChart(barChartRef.current);
  };

  const current = gamma(n, d);
  const growth = d > 1 ? (1 + d) : 1;
  const orders = Math.log10(current);

  return (
    <div className="flex h-full bg-gray-50">
      {/* LEFT PANE */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-6 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-xl font-serif">Po<span className="italic text-red-600">S</span>L</h1>
          <p className="text-xs text-gray-500">Composition Explosion</p>
        </div>

        {/* Composition Depth */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Composition Depth (n)
          </h3>
          <div className="flex justify-between text-xs mb-2">
            <span className="font-bold">Evidence observations:</span>
            <span className="font-mono font-bold text-red-600">{n}</span>
          </div>
          <input
            type="range"
            min="1"
            max="9"
            value={n}
            onChange={(e) => setN(parseInt(e.target.value))}
            className="w-full h-1 bg-gradient-to-r from-green-600 to-red-600 rounded appearance-none cursor-pointer"
          />
        </div>

        {/* Channels */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Channels (d)
          </h3>
          <div className="flex justify-between text-xs mb-2">
            <span className="font-bold">Independent modalities:</span>
            <span className="font-mono font-bold text-red-600">{d}</span>
          </div>
          <input
            type="range"
            min="1"
            max="8"
            value={d}
            onChange={(e) => setD(parseInt(e.target.value))}
            className="w-full h-1 bg-gradient-to-r from-green-600 to-red-600 rounded appearance-none cursor-pointer"
          />
        </div>

        {/* Formula */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Formula
          </h3>
          <div className="bg-gray-50 border border-gray-200 rounded p-3 text-xs font-mono space-y-1">
            <div>Γ(n,d) = d(1+d)^(n-1)</div>
            <div className="text-gray-500 text-[11px]">Distinguishable evidence tuples</div>
          </div>
        </div>

        {/* Current State */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Current State
          </h3>

          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">State Space Size</div>
            <div className="text-xl font-mono font-bold text-red-600">{Math.floor(current).toLocaleString()}</div>
            <div className="text-xs text-gray-500 mt-1">Total distinguishable position hypotheses</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Growth per depth</div>
            <div className="text-xl font-mono font-bold text-red-600">{growth.toFixed(1)}x</div>
            <div className="text-xs text-gray-500 mt-1">Multiplicative factor from n to n+1</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Orders of magnitude</div>
            <div className="text-xl font-mono font-bold text-red-600">{orders.toFixed(2)}</div>
            <div className="text-xs text-gray-500 mt-1">log₁₀(Γ(n,d))</div>
          </div>
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex flex-col gap-4">
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <canvas ref={lineChartRef} className="w-full h-full" />
        </div>
        <div className="flex gap-4 h-1/3">
          <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
            <canvas ref={heatmapRef} className="w-full h-full" />
          </div>
          <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
            <canvas ref={barChartRef} className="w-full h-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
