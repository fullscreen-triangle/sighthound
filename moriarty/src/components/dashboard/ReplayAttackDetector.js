import React, { useState, useEffect, useRef } from 'react';

export default function ReplayAttackDetector() {
  const [epoch, setEpoch] = useState(0);
  const [hasRecorded, setHasRecorded] = useState(false);
  const [delta, setDelta] = useState(1);
  const detectionChartRef = useRef(null);
  const driftChartRef = useRef(null);

  const computeDetectionRate = (d) => {
    const baseDetection = [
      0.13, 0.21, 0.32, 0.45, 0.58,
      0.70, 0.80, 0.87, 0.92, 0.95,
      0.96, 0.97, 0.975, 0.98, 0.985,
      0.99, 0.992, 0.994, 0.995, 0.996
    ];
    return baseDetection[Math.min(d - 1, 19)];
  };

  const recordEvidence = () => {
    setEpoch(Math.floor(Date.now() / 1000) % 10000);
    setHasRecorded(true);
  };

  const replayAttack = () => {
    if (!hasRecorded) {
      alert('Record an observation first!');
      return;
    }

    const detectionRate = computeDetectionRate(delta);
    const detected = Math.random() < detectionRate;
    const drift = Math.exp(-0.6 * delta);

    const message = detected
      ? `[DETECTED] Replay attempt at epoch E'=${epoch + delta} rejected.\nOld evidence at E=${epoch} caused likelihood drift by factor ${drift.toFixed(3)}.`
      : `[EVADED] Replay succeeded at E'=${epoch + delta}. (This is rare when δ > 5)`;

    alert(message);
  };

  useEffect(() => {
    drawCharts();
  }, [delta, hasRecorded]);

  const drawDetectionChart = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const padding = 40;
    const graphWidth = canvas.width - 2 * padding;
    const graphHeight = canvas.height - 2 * padding;

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
    for (let i = 0; i <= 10; i++) {
      const y = canvas.height - padding - (i / 10) * graphHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();
    }

    // Draw detection curve
    ctx.strokeStyle = '#c85a54';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let d = 1; d <= 20; d++) {
      const detRate = computeDetectionRate(d);
      const x = padding + ((d - 1) / 19) * graphWidth;
      const y = canvas.height - padding - detRate * graphHeight;

      if (d === 1) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Highlight current delta
    if (hasRecorded) {
      const detRate = computeDetectionRate(delta);
      const x = padding + ((delta - 1) / 19) * graphWidth;
      const y = canvas.height - padding - detRate * graphHeight;

      ctx.fillStyle = 'rgba(200, 90, 84, 0.2)';
      ctx.fillRect(x - 6, y - 6, 12, 12);
      ctx.fillStyle = '#c85a54';
      ctx.fillRect(x - 3, y - 3, 6, 6);
    }

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 20; i += 5) {
      const x = padding + (i / 19) * graphWidth;
      ctx.fillText(i, x, canvas.height - 20);
    }
    ctx.fillText('Epoch Delta (δ)', canvas.width / 2, canvas.height - 5);

    // Y-axis labels
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 10; i += 2) {
      const y = canvas.height - padding - (i / 10) * graphHeight;
      ctx.fillText((i * 10) + '%', padding - 10, y);
    }
  };

  const drawDriftChart = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const padding = 40;
    const barWidth = (canvas.width - 2 * padding) / 20;

    for (let d = 1; d <= 20; d++) {
      const drift = Math.exp(-0.6 * d);
      const x = padding + (d - 1) * barWidth;
      const height = drift * (canvas.height - 2 * padding);
      const y = canvas.height - padding - height;

      const color = delta === d ? '#c85a54' : '#ddd';
      ctx.fillStyle = color;
      ctx.fillRect(x + 2, y, barWidth - 4, height);

      ctx.strokeStyle = '#999';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x + 2, y, barWidth - 4, height);
    }

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

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 1; i <= 20; i += 5) {
      const x = padding + (i - 1) * barWidth + barWidth / 2;
      ctx.fillText(i, x, canvas.height - 20);
    }
    ctx.fillText('Epoch Delta (δ)', canvas.width / 2, canvas.height - 5);

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = '10px Georgia';
    for (let i = 0; i <= 10; i += 2) {
      const y = canvas.height - padding - (i / 10) * (canvas.height - 2 * padding);
      ctx.fillText((i / 10).toFixed(1), padding - 10, y);
    }
  };

  const drawCharts = () => {
    drawDetectionChart(detectionChartRef.current);
    drawDriftChart(driftChartRef.current);
  };

  const detRate = hasRecorded ? computeDetectionRate(delta) : null;
  const evasionRate = detRate ? (1 - detRate) : null;
  const drift = Math.exp(-0.6 * delta);

  let explanation = hasRecorded
    ? delta <= 3
      ? `At δ=${delta}, the detection rate is only ${(detRate * 100).toFixed(0)}%. Low epoch offset allows some evasion, but monotone epoch counters still attenuate the replayed evidence by factor ${drift.toFixed(3)}.`
      : delta <= 10
      ? `At δ=${delta}, the detection rate jumps to ${(detRate * 100).toFixed(0)}%. The signature's epoch conditioning causes sharp drift: exp(-0.6·${delta}) = ${drift.toFixed(3)}. Replayed observations now land in different position cells.`
      : `At δ=${delta}, detection is nearly certain (${(detRate * 100).toFixed(0)}%). Monotone epoch counters provide structural replay immunity: timing deviations grow exponentially with epoch offset, making old observations structurally incompatible with current epoch.`
    : "Record an observation at the current epoch first. Then try replaying it at a future epoch.";

  return (
    <div className="flex h-full bg-gray-50">
      {/* LEFT PANE */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-6 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-xl font-serif">Po<span className="italic text-red-600">S</span>L</h1>
          <p className="text-xs text-gray-500">Replay Attack Detector</p>
        </div>

        {/* Original Evidence */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Original Evidence
          </h3>
          <div className="bg-gray-50 border border-gray-200 rounded p-3 text-center mb-3">
            <div className="text-xs text-gray-500 uppercase font-bold mb-1">Epoch E</div>
            <div className="font-mono font-bold text-lg text-red-600">{hasRecorded ? epoch : '-'}</div>
          </div>
          <button
            onClick={recordEvidence}
            className="w-full px-3 py-2 bg-green-600 text-white rounded font-bold text-xs hover:bg-green-700"
          >
            Record Observation
          </button>
        </div>

        {/* Replay Attack */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Replay Attack
          </h3>
          <div className="mb-3">
            <div className="flex justify-between text-xs mb-2">
              <span className="font-bold">Epoch Delta (δ):</span>
              <span className="font-mono font-bold text-red-600">{delta}</span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={delta}
              onChange={(e) => setDelta(parseInt(e.target.value))}
              className="w-full h-1 bg-gradient-to-r from-green-600 to-red-600 rounded appearance-none cursor-pointer"
            />
          </div>
          <button
            onClick={replayAttack}
            disabled={!hasRecorded}
            className={`w-full px-3 py-2 rounded font-bold text-xs ${
              hasRecorded
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
          >
            Replay at E'=E+δ
          </button>
        </div>

        {/* Metrics */}
        <div className="mb-6">
          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Detection Rate</div>
            <div className="text-xl font-mono font-bold text-red-600">
              {hasRecorded ? (detRate * 100).toFixed(1) + '%' : '-'}
            </div>
            <div className="text-xs text-gray-500 mt-1">Probability replay is detected</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Evasion Probability</div>
            <div className="text-xl font-mono font-bold text-red-600">
              {hasRecorded ? (evasionRate * 100).toFixed(1) + '%' : '-'}
            </div>
            <div className="text-xs text-gray-500 mt-1">P(attack succeeds undetected)</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Likelihood Drift</div>
            <div className="text-xl font-mono font-bold text-red-600">{drift.toFixed(3)}</div>
            <div className="text-xs text-gray-500 mt-1">exp(-0.6·δ) attenuation</div>
          </div>
        </div>

        {/* Explanation */}
        <div className="mb-6">
          <div className="bg-gray-50 border-l-4 border-red-600 p-3 rounded text-xs text-gray-700 leading-relaxed">
            {explanation}
          </div>
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex flex-col gap-4">
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <canvas ref={detectionChartRef} className="w-full h-full" />
        </div>
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <canvas ref={driftChartRef} className="w-full h-full" />
        </div>
      </div>
    </div>
  );
}
