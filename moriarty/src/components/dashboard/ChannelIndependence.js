import React, { useState, useEffect, useRef } from 'react';

const channels = [
  { id: 'video', name: 'Video', emoji: '📹' },
  { id: 'audio', name: 'Audio', emoji: '🔊' },
  { id: 'motion', name: 'Motion', emoji: '🎬' },
  { id: 'temp', name: 'Temperature', emoji: '🌡️' },
  { id: 'pressure', name: 'Pressure', emoji: '🔵' },
  { id: 'humidity', name: 'Humidity', emoji: '💧' }
];

export default function ChannelIndependence() {
  const [activeChannels, setActiveChannels] = useState([true, false, false, false, false, false]);
  const chartRef = useRef(null);

  const regionA = { video: 0.6, audio: 0.4, motion: 0.8, temp: 0.5, pressure: 0.3, humidity: 0.7 };
  const regionB = { video: 0.5, audio: 0.4, motion: 0.2, temp: 0.6, pressure: 0.7, humidity: 0.3 };

  const computeCollisionProbability = () => {
    let prob = 1.0;
    for (let i = 0; i < channels.length; i++) {
      if (activeChannels[i]) {
        const chid = channels[i].id;
        const diff = Math.abs(regionA[chid] - regionB[chid]);
        prob *= (1 - diff);
      }
    }
    return Math.max(0, prob);
  };

  const toggleChannel = (idx) => {
    if (idx === 0) return; // First channel always active
    const newActive = [...activeChannels];
    newActive[idx] = !newActive[idx];
    setActiveChannels(newActive);
  };

  useEffect(() => {
    drawChart();
  }, [activeChannels]);

  const drawChart = () => {
    const canvas = chartRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 32;
    canvas.height = rect.height - 32;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const activeCount = activeChannels.filter(x => x).length;
    const padding = 50;
    const barWidth = (canvas.width - 2 * padding) / channels.length;

    // Calculate collision probabilities for each subset
    const probs = [];
    for (let mask = 1; mask < Math.pow(2, channels.length); mask++) {
      let prob = 1.0;
      for (let i = 0; i < channels.length; i++) {
        if ((mask & (1 << i)) !== 0) {
          const chid = channels[i].id;
          const diff = Math.abs(regionA[chid] - regionB[chid]);
          prob *= (1 - diff);
        }
      }
      const count = countBits(mask);
      probs.push({ mask, prob, count });
    }

    const byCount = {};
    probs.forEach(p => {
      if (!byCount[p.count]) byCount[p.count] = [];
      byCount[p.count].push(p.prob);
    });

    const countValues = Object.keys(byCount)
      .map(c => ({
        count: parseInt(c),
        median: getMedian(byCount[c]),
        max: Math.max(...byCount[c])
      }))
      .sort((a, b) => a.count - b.count);

    const barHeight = canvas.height - 2 * padding;

    countValues.forEach((item, idx) => {
      const x = padding + idx * barWidth;

      const medianHeight = (1 - item.median) * barHeight;
      const maxHeight = (1 - item.max) * barHeight;

      ctx.fillStyle = activeChannels[item.count] ? '#c85a54' : '#ddd';
      ctx.fillRect(x + 8, canvas.height - padding - medianHeight, barWidth - 16, medianHeight);

      ctx.strokeStyle = '#999';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2 - 8, canvas.height - padding - maxHeight);
      ctx.lineTo(x + barWidth / 2 + 8, canvas.height - padding - maxHeight);
      ctx.stroke();

      ctx.fillStyle = '#333';
      ctx.font = 'bold 12px Georgia';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(item.count, x + barWidth / 2, canvas.height - 20);
    });

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

    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Channels', canvas.width / 2, canvas.height - 5);
  };

  const countBits = (n) => {
    let count = 0;
    while (n) {
      count += n & 1;
      n >>= 1;
    }
    return count;
  };

  const getMedian = (arr) => {
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted[mid];
  };

  const collProb = computeCollisionProbability();
  const activeCount = activeChannels.filter(x => x).length;
  const activeList = channels.filter((ch, i) => activeChannels[i]).map(ch => ch.name).join(', ');

  let explanation = '';
  if (collProb > 0.9) {
    explanation = `With only <strong>${activeCount} channel(s)</strong>, the two regions produce nearly identical signatures. The system cannot distinguish between them.`;
  } else if (collProb > 0.5) {
    explanation = `With <strong>${activeCount} channels</strong>, there is still significant ambiguity (${(collProb * 100).toFixed(1)}% collision probability). Adding more independent channels will sharpen the distinction.`;
  } else if (collProb > 0.1) {
    explanation = `With <strong>${activeCount} channels</strong>, the collision probability has dropped to ${(collProb * 100).toFixed(1)}%. The regions are becoming distinguishable.`;
  } else {
    explanation = `With <strong>${activeCount} channels</strong>, the regions are now clearly distinguishable. Multi-modal fusion via product likelihood breaks all ambiguity: Pr[Ω|τ] ∝ ∏ σ(c).`;
  }

  return (
    <div className="flex h-full bg-gray-50">
      {/* LEFT PANE */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-6 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-xl font-serif">Po<span className="italic text-red-600">S</span>L</h1>
          <p className="text-xs text-gray-500">Channel Independence</p>
        </div>

        {/* Add Channels */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Add Channels
          </h3>
          <div className="flex flex-col gap-2">
            {channels.map((ch, idx) => (
              <button
                key={ch.id}
                onClick={() => toggleChannel(idx)}
                disabled={idx === 0}
                className={`px-3 py-2 text-xs rounded text-center font-bold transition ${
                  activeChannels[idx]
                    ? 'bg-red-600 text-white border border-red-600'
                    : idx === 0
                    ? 'bg-gray-100 text-gray-400 border border-gray-300 cursor-not-allowed'
                    : 'bg-white text-gray-700 border border-gray-300 hover:border-gray-500'
                }`}
              >
                {ch.emoji} {ch.name}
              </button>
            ))}
          </div>
        </div>

        {/* Metrics */}
        <div className="mb-6">
          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Active Channels</div>
            <div className="text-xl font-mono font-bold text-red-600">{activeCount}</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3 mb-2">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Collision Probability</div>
            <div className="text-xl font-mono font-bold text-red-600">{collProb.toFixed(3)}</div>
            <div className="text-xs text-gray-500 mt-1">P(Region A signature = Region B signature)</div>
          </div>

          <div className="bg-white border border-gray-200 rounded p-3">
            <div className="text-xs font-bold uppercase text-gray-600 mb-1">Ambiguity</div>
            <div className="text-xl font-mono font-bold text-red-600">{(collProb * 100).toFixed(1)}%</div>
            <div className="text-xs text-gray-500 mt-1">Indistinguishable regions</div>
          </div>
        </div>

        {/* Explanation */}
        <div className="mb-6">
          <div className="bg-gray-50 border-l-4 border-red-600 p-3 rounded text-xs text-gray-700 leading-relaxed">
            <div dangerouslySetInnerHTML={{ __html: explanation }} />
          </div>
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex flex-col gap-4">
        {/* Region comparison */}
        <div className="flex-1 bg-white border border-gray-200 rounded p-6 overflow-hidden">
          <div className="flex gap-8 items-flex-end justify-center h-full">
            {/* Region A */}
            <div className="flex flex-col items-center gap-3">
              <div className={`text-xs font-bold ${collProb > 0.5 ? 'text-gray-400' : 'text-gray-700'}`}>
                Region A
              </div>
              <div className="flex items-flex-end gap-1 h-40">
                {activeChannels.map((active, idx) => {
                  if (!active) return null;
                  const ch = channels[idx];
                  const val = regionA[ch.id];
                  return (
                    <div
                      key={ch.id}
                      className={`w-6 ${collProb > 0.5 ? 'bg-gray-400' : 'bg-red-600'} rounded-t flex items-flex-end justify-center text-white text-xs font-bold`}
                      style={{ height: val * 160 + 'px' }}
                    >
                      {ch.emoji}
                    </div>
                  );
                })}
              </div>
              <div className="bg-gray-50 rounded p-2 text-xs text-gray-600">{activeList}</div>
            </div>

            {/* Divider */}
            <div className="w-px bg-gray-300 h-full" />

            {/* Region B */}
            <div className="flex flex-col items-center gap-3">
              <div className={`text-xs font-bold ${collProb > 0.5 ? 'text-gray-400' : 'text-gray-700'}`}>
                Region B
              </div>
              <div className="flex items-flex-end gap-1 h-40">
                {activeChannels.map((active, idx) => {
                  if (!active) return null;
                  const ch = channels[idx];
                  const val = regionB[ch.id];
                  return (
                    <div
                      key={ch.id}
                      className={`w-6 ${collProb > 0.5 ? 'bg-gray-400' : 'bg-red-600'} rounded-t flex items-flex-end justify-center text-white text-xs font-bold`}
                      style={{ height: val * 160 + 'px' }}
                    >
                      {ch.emoji}
                    </div>
                  );
                })}
              </div>
              <div className="bg-gray-50 rounded p-2 text-xs text-gray-600">{activeList}</div>
            </div>
          </div>
        </div>

        {/* Probability chart */}
        <div className="bg-white border border-gray-200 rounded p-4 overflow-hidden h-48">
          <canvas ref={chartRef} className="w-full h-full" />
        </div>
      </div>
    </div>
  );
}
