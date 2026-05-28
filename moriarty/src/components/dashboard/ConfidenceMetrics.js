import React, { useState, useEffect, useRef } from 'react';

const patterns = [
  { name: 'Random', desc: 'Uniformly random evidence' },
  { name: 'Biased', desc: 'Concentrated on true region' },
  { name: 'Adversarial', desc: 'Attacking true region' }
];

export default function ConfidenceMetrics() {
  const [pattern, setPattern] = useState(0);
  const [evidence, setEvidence] = useState([]);
  const [margin, setMargin] = useState([]);
  const [strength, setStrength] = useState([]);
  const [floor, setFloor] = useState([]);
  const timeSeriesChartRef = useRef(null);
  const jointChartRef = useRef(null);

  const simulateEvidence = (p) => {
    const trueRegion = 2;
    const totalRegions = 9;

    if (p === 0) {
      return Math.floor(Math.random() * totalRegions);
    } else if (p === 1) {
      return Math.random() < 0.7 ? trueRegion : Math.floor(Math.random() * totalRegions);
    } else {
      return Math.random() < 0.6 ? (trueRegion + 3) % totalRegions : Math.floor(Math.random() * totalRegions);
    }
  };

  const computeMetrics = (evidenceList) => {
    const totalRegions = 9;

    if (evidenceList.length === 0) {
      return { margin: [], strength: [], floor: [] };
    }

    const marginList = [];
    const strengthList = [];
    const floorList = [];

    let likelihoods = Array(totalRegions).fill(1);

    evidenceList.forEach((regionIdx, stepIdx) => {
      for (let i = 0; i < totalRegions; i++) {
        const dist = Math.abs(i - regionIdx);
        const similarity = Math.exp(-0.3 * dist);
        likelihoods[i] *= similarity;
      }

      const total = likelihoods.reduce((a, b) => a + b);
      const posteriors = likelihoods.map(l => l / total);

      const sorted = [...posteriors].sort((a, b) => b - a);
      marginList.push(sorted[0] - sorted[1]);
      strengthList.push(sorted[0]);

      const d = 3;
      const n = stepIdx + 1;
      const gamma = d * Math.pow(1 + d, n - 1);
      floorList.push(1 / gamma);
    });

    return { margin: marginList, strength: strengthList, floor: floorList };
  };

  const addEvidence = (count) => {
    const newEvidence = [...evidence];
    for (let i = 0; i < count; i++) {
      newEvidence.push(simulateEvidence(pattern));
    }
    setEvidence(newEvidence);

    const metrics = computeMetrics(newEvidence);
    setMargin(metrics.margin);
    setStrength(metrics.strength);
    setFloor(metrics.floor);
  };

  const resetEvidence = () => {
    setEvidence([]);
    setMargin([]);
    setStrength([]);
    setFloor([]);
  };

  useEffect(() => {
    drawCharts();
  }, [margin, strength, floor]);

  const drawTimeSeriesChart = (canvas) => {
    if (!canvas || evidence.length === 0) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (evidence.length === 0) {
      ctx.fillStyle = '#999';
      ctx.font = '12px Georgia';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Add evidence to see metrics evolve', canvas.width / 2, canvas.height / 2);
      return;
    }

    const padding = 40;
    const graphWidth = canvas.width - 2 * padding;
    const graphHeight = canvas.height - 2 * padding;

    // Draw grid
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = canvas.height - padding - (i / 5) * graphHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();
    }

    const datasets = [
      { data: margin, color: '#c85a54', name: 'Margin' },
      { data: strength, color: '#2d5016', name: 'Strength' },
      { data: floor, color: '#d4a574', name: 'Floor' }
    ];

    datasets.forEach((dataset) => {
      ctx.strokeStyle = dataset.color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const max = 1.0;

      dataset.data.forEach((val, idx) => {
        const x = padding + (idx / evidence.length) * graphWidth;
        const y = canvas.height - padding - (val / max) * graphHeight;

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    });

    // Axes
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

    // Legend
    ctx.fillStyle = '#666';
    ctx.font = '10px Georgia';
    ctx.textAlign = 'left';
    datasets.forEach((ds, idx) => {
      ctx.fillStyle = ds.color;
      ctx.fillRect(10, 10 + idx * 16, 10, 10);
      ctx.fillStyle = '#333';
      ctx.fillText(ds.name, 25, 15 + idx * 16);
    });

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Evidence Count (n)', canvas.width / 2, canvas.height - 5);
  };

  const drawJointChart = (canvas) => {
    if (!canvas || evidence.length === 0) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (evidence.length === 0) {
      ctx.fillStyle = '#999';
      ctx.font = '12px Georgia';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Margin vs Strength', canvas.width / 2, canvas.height / 2 - 10);
      return;
    }

    const padding = 50;
    const graphWidth = canvas.width - 2 * padding;
    const graphHeight = canvas.height - 2 * padding;

    const maxMargin = Math.max(...margin);
    const maxStrength = Math.max(...strength);

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

    // Draw points
    margin.forEach((m, idx) => {
      const s = strength[idx];
      const f = floor[idx];

      const x = padding + (m / maxMargin) * graphWidth;
      const y = canvas.height - padding - (s / maxStrength) * graphHeight;

      const intensity = Math.min(1 - f * 100, 1);
      ctx.fillStyle = `rgba(200, 90, 84, ${0.3 + intensity * 0.4})`;
      ctx.fillRect(x - 4, y - 4, 8, 8);

      ctx.strokeStyle = `rgba(200, 90, 84, ${0.6 + intensity * 0.4})`;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 4, y - 4, 8, 8);
    });

    // Highlight latest
    if (margin.length > 0) {
      const lastIdx = margin.length - 1;
      const x = padding + (margin[lastIdx] / maxMargin) * graphWidth;
      const y = canvas.height - padding - (strength[lastIdx] / maxStrength) * graphHeight;
      ctx.strokeStyle = '#c85a54';
      ctx.lineWidth = 3;
      ctx.strokeRect(x - 6, y - 6, 12, 12);
    }

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '10px Georgia';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Posterior Margin', canvas.width / 2, canvas.height - 10);
  };

  const drawCharts = () => {
    drawTimeSeriesChart(timeSeriesChartRef.current);
    drawJointChart(jointChartRef.current);
  };

  const currentMargin = margin.length > 0 ? margin[margin.length - 1] : 0;
  const currentStrength = strength.length > 0 ? strength[strength.length - 1] : 0;
  const currentFloor = floor.length > 0 ? floor[floor.length - 1] : 0;

  return (
    <div className="flex h-full bg-gray-50">
      {/* LEFT PANE */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-6 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-xl font-serif">Po<span className="italic text-red-600">S</span>L</h1>
          <p className="text-xs text-gray-500">Confidence Metrics</p>
        </div>

        {/* Pattern selector */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Evidence Pattern
          </h3>
          <div className="flex flex-wrap gap-2">
            {patterns.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setPattern(idx);
                  resetEvidence();
                }}
                className={`px-2 py-1 text-xs rounded-full transition ${
                  pattern === idx
                    ? 'bg-red-600 text-white border border-red-600'
                    : 'bg-white border border-gray-300 hover:border-gray-500'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Evidence counter */}
        <div className="mb-6 bg-gray-50 border border-gray-200 rounded p-3 text-center font-mono font-bold text-lg">
          <div className="text-xs text-gray-500 uppercase mb-1">n</div>
          <div className="text-red-600">{evidence.length}</div>
        </div>

        {/* Controls */}
        <div className="mb-6 flex flex-col gap-2">
          <button
            onClick={() => addEvidence(1)}
            className="px-3 py-2 bg-red-600 text-white rounded font-bold text-xs hover:bg-red-700"
          >
            Add Evidence (+1)
          </button>
          <button
            onClick={() => addEvidence(5)}
            className="px-3 py-2 bg-red-600 text-white rounded font-bold text-xs hover:bg-red-700"
          >
            Batch (+5)
          </button>
          <button
            onClick={resetEvidence}
            className="px-3 py-2 bg-white border border-gray-300 rounded font-bold text-xs hover:bg-gray-50"
          >
            Reset
          </button>
        </div>

        {/* Posterior Margin */}
        <div className="mb-4">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-2 pb-2 border-b border-gray-200">
            Posterior Margin
          </h3>
          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xl font-mono font-bold text-red-600">{currentMargin.toFixed(3)}</div>
            <div className="h-1 bg-gray-200 rounded mt-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-600 to-red-600 transition-all"
                style={{ width: Math.min(currentMargin * 200, 100) + '%' }}
              />
            </div>
            <div className="text-xs text-gray-600 mt-2 leading-relaxed">
              <strong>Sharpness of belief.</strong> Difference between top and 2nd-best region.
            </div>
            <div className="text-xs font-mono text-red-600 mt-2">max Pr[Ω|τ] − 2nd max</div>
          </div>
        </div>

        {/* Signature Strength */}
        <div className="mb-4">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-2 pb-2 border-b border-gray-200">
            Signature Strength
          </h3>
          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xl font-mono font-bold text-red-600">{currentStrength.toFixed(3)}</div>
            <div className="h-1 bg-gray-200 rounded mt-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-600 to-red-600 transition-all"
                style={{ width: currentStrength * 100 + '%' }}
              />
            </div>
            <div className="text-xs text-gray-600 mt-2 leading-relaxed">
              <strong>Typicality of evidence.</strong> Sum of winner's signature probabilities.
            </div>
            <div className="text-xs font-mono text-red-600 mt-2">Σ σ(Ω*)(c)</div>
          </div>
        </div>

        {/* Composition Floor */}
        <div className="mb-4">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-2 pb-2 border-b border-gray-200">
            Composition Floor
          </h3>
          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xl font-mono font-bold text-red-600">{currentFloor.toFixed(4)}</div>
            <div className="h-1 bg-gray-200 rounded mt-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-600 to-red-600 transition-all"
                style={{ width: Math.max(currentFloor * 100, 1) + '%' }}
              />
            </div>
            <div className="text-xs text-gray-600 mt-2 leading-relaxed">
              <strong>Theoretical minimum ambiguity.</strong> State space constraint.
            </div>
            <div className="text-xs font-mono text-red-600 mt-2">1 / Γ(n,d)</div>
          </div>
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex flex-col gap-4">
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <canvas ref={timeSeriesChartRef} className="w-full h-full" />
        </div>
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <canvas ref={jointChartRef} className="w-full h-full" />
        </div>
      </div>
    </div>
  );
}
