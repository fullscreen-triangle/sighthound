import React, { useState, useEffect } from 'react';

const scenarios = {
  building: {
    name: "Building",
    regions: ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5", "Room 6", "Room 7", "Room 8", "Room 9"],
    channels: [
      { id: "video", name: "Video", cells: ["wall", "floor", "door"] },
      { id: "audio", name: "Audio", cells: ["quiet", "medium", "loud"] }
    ]
  },
  warehouse: {
    name: "Warehouse",
    regions: ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E", "Zone F", "Zone G", "Zone H", "Zone I"],
    channels: [
      { id: "rfid", name: "RFID", cells: ["near", "far"] },
      { id: "temp", name: "Temperature", cells: ["cold", "warm", "hot"] },
      { id: "motion", name: "Motion", cells: ["static", "moving"] }
    ]
  },
  field: {
    name: "Sports Field",
    regions: ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
    channels: [
      { id: "camera1", name: "Cam 1", cells: ["visible", "hidden"] },
      { id: "camera2", name: "Cam 2", cells: ["visible", "hidden"] }
    ]
  }
};

function generateSignatures(scenario) {
  const sigs = {};
  for (let region of scenario.regions) {
    sigs[region] = {};
    for (let channel of scenario.channels) {
      const probs = {};
      let sum = 0;
      for (let cell of channel.cells) {
        probs[cell] = Math.random();
        sum += probs[cell];
      }
      for (let cell of channel.cells) {
        probs[cell] /= sum;
      }
      sigs[region][channel.id] = probs;
    }
  }
  return sigs;
}

export default function EvidenceAccumulator() {
  const [currentScenario, setCurrentScenario] = useState("building");
  const [scenario, setScenario] = useState(scenarios.building);
  const [signatures, setSignatures] = useState(generateSignatures(scenarios.building));
  const [evidence, setEvidence] = useState([]);
  const [posteriors, setPosteriori] = useState({});

  useEffect(() => {
    const newScenario = scenarios[currentScenario];
    setScenario(newScenario);
    setSignatures(generateSignatures(newScenario));
    setEvidence([]);
    setPosteriori({});
  }, [currentScenario]);

  const computePosteriorsFromEvidence = (evidenceList) => {
    if (evidenceList.length === 0) {
      const uniform = 1.0 / scenario.regions.length;
      const priors = {};
      scenario.regions.forEach(r => (priors[r] = uniform));
      return priors;
    }

    const likelihoods = {};
    scenario.regions.forEach(region => {
      let likelihood = 1.0;
      evidenceList.forEach(obs => {
        const sig = signatures[region][obs.channel];
        if (sig && sig[obs.cell]) {
          likelihood *= sig[obs.cell];
        }
      });
      likelihoods[region] = likelihood;
    });

    const total = Object.values(likelihoods).reduce((a, b) => a + b, 0);
    const post = {};
    scenario.regions.forEach(r => {
      post[r] = total > 0 ? likelihoods[r] / total : 1.0 / scenario.regions.length;
    });
    return post;
  };

  const addEvidence = (channelId, cell) => {
    const newEvidence = [...evidence, { channel: channelId, cell, timestamp: Date.now() }];
    setEvidence(newEvidence);
    setPosteriori(computePosteriorsFromEvidence(newEvidence));
  };

  const clearEvidence = () => {
    setEvidence([]);
    setPosteriori({});
  };

  const addRandomEvidence = () => {
    const trueRegion = scenario.regions[Math.floor(Math.random() * scenario.regions.length)];
    for (let i = 0; i < 5; i++) {
      const channel = scenario.channels[Math.floor(Math.random() * scenario.channels.length)];
      const sig = signatures[trueRegion][channel.id];
      const cells = Object.keys(sig);
      const cell = cells[Math.floor(Math.random() * cells.length)];
      addEvidence(channel.id, cell);
    }
  };

  const winner = scenario.regions.reduce(
    (max, r) => (posteriors[r] > (posteriors[max] || 0)) ? r : max,
    null
  );

  const posteriorValues = scenario.regions.map(r => posteriors[r] || 0).sort((a, b) => b - a);
  const margin = posteriorValues[0] - (posteriorValues[1] || 0);
  const strength = winner ? Object.values(signatures[winner]).reduce((sum, channelSig) =>
    sum + Object.values(channelSig).reduce((a, b) => a + b, 0), 0) / scenario.channels.length : 0;
  const gamma = scenario.channels.length * Math.pow(1 + scenario.channels.length, Math.max(0, evidence.length - 1));

  const sorted = scenario.regions
    .map(r => ({ region: r, prob: posteriors[r] || 0 }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 6);

  return (
    <div className="flex h-full bg-gray-50">
      {/* LEFT PANE */}
      <div className="w-1/4 bg-white border-r border-gray-200 p-6 overflow-y-auto">
        <div className="mb-6">
          <h1 className="text-xl font-serif">Po<span className="italic text-red-600">S</span>L</h1>
          <p className="text-xs text-gray-500">Evidence Accumulator</p>
        </div>

        {/* Scenario selector */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Scenario
          </h3>
          <div className="flex flex-wrap gap-2">
            {Object.keys(scenarios).map(key => (
              <button
                key={key}
                onClick={() => setCurrentScenario(key)}
                className={`px-3 py-1 text-xs rounded-full transition ${
                  currentScenario === key
                    ? 'bg-red-600 text-white border border-red-600'
                    : 'bg-white border border-gray-300 hover:border-gray-500'
                }`}
              >
                {scenarios[key].name}
              </button>
            ))}
          </div>
        </div>

        {/* Partition display */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Partition (P)
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {scenario.regions.map(region => (
              <div
                key={region}
                className={`p-2 text-xs text-center rounded border transition ${
                  winner === region
                    ? 'bg-red-600 text-white border-red-600 font-bold'
                    : 'bg-white border-gray-300'
                }`}
              >
                {region}
              </div>
            ))}
          </div>
        </div>

        {/* Status strip */}
        <div className="grid grid-cols-3 gap-2 p-3 bg-gray-50 border border-gray-200 rounded mb-6 text-xs">
          <div>
            <div className="text-gray-500 font-bold uppercase text-[10px]">Depth</div>
            <div className="font-mono font-bold text-red-600">{evidence.length}</div>
          </div>
          <div>
            <div className="text-gray-500 font-bold uppercase text-[10px]">Channels</div>
            <div className="font-mono font-bold text-red-600">{scenario.channels.length}</div>
          </div>
          <div>
            <div className="text-gray-500 font-bold uppercase text-[10px]">Winner</div>
            <div className="font-mono font-bold text-red-600">{winner || '-'}</div>
          </div>
        </div>

        {/* Channel controls */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-3 pb-2 border-b border-gray-200">
            Add Evidence
          </h3>
          {scenario.channels.map(channel => (
            <div key={channel.id} className="mb-3">
              <div className="text-xs font-bold text-gray-600 mb-2">{channel.name}</div>
              <div className="grid grid-cols-2 gap-1">
                {channel.cells.map(cell => (
                  <button
                    key={cell}
                    onClick={() => addEvidence(channel.id, cell)}
                    className="px-2 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50"
                  >
                    {cell}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <button
            onClick={clearEvidence}
            className="w-full mt-3 px-3 py-2 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50 font-bold"
          >
            Clear All
          </button>
          <button
            onClick={addRandomEvidence}
            className="w-full mt-2 px-3 py-2 text-xs bg-red-600 text-white rounded hover:bg-red-700 font-bold"
          >
            Random Burst (5)
          </button>
        </div>

        {/* Evidence log */}
        <div className="mb-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-700 mb-2 pb-2 border-b border-gray-200">
            Evidence Log (τ)
          </h3>
          <div className="border border-gray-200 rounded p-2 max-h-40 overflow-y-auto bg-white text-xs">
            {evidence.length === 0 ? (
              <div className="text-gray-400">No evidence yet</div>
            ) : (
              evidence.map((obs, idx) => (
                <div key={idx} className="py-1 text-gray-600 font-mono">
                  <span className="text-gray-400">[{idx}]</span> {obs.channel} → {obs.cell}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 p-6 flex flex-col gap-4">
        {/* Map */}
        <div className="flex-1 bg-white border border-gray-200 rounded p-4 overflow-hidden">
          <div className="grid grid-cols-3 gap-3 h-full">
            {scenario.regions.map(region => {
              const prob = posteriors[region] || 0;
              const intensity = Math.min(prob * 3, 1);
              return (
                <div
                  key={region}
                  className="border-2 rounded-lg flex flex-col items-center justify-center transition"
                  style={{
                    backgroundColor: `rgba(200, 90, 84, ${intensity * 0.3})`,
                    borderColor: `rgba(200, 90, 84, ${0.2 + intensity * 0.8})`
                  }}
                >
                  <div className="text-xs font-bold text-gray-700">{region}</div>
                  <div className="text-lg font-bold text-red-600">{(prob * 100).toFixed(1)}%</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Metrics and rankings */}
        <div className="flex gap-4">
          {/* Metrics */}
          <div className="w-64 flex flex-col gap-3">
            <div className="bg-white border border-gray-200 rounded p-3">
              <div className="text-xs font-bold uppercase text-gray-600 mb-2">Posterior Margin</div>
              <div className="text-xl font-mono font-bold text-red-600">{margin.toFixed(3)}</div>
              <div className="h-1 bg-gray-200 rounded mt-2">
                <div
                  className="h-full bg-gradient-to-r from-green-600 to-red-600 rounded"
                  style={{ width: Math.min(margin * 200, 100) + '%' }}
                />
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded p-3">
              <div className="text-xs font-bold uppercase text-gray-600 mb-2">Signature Strength</div>
              <div className="text-xl font-mono font-bold text-red-600">{strength.toFixed(3)}</div>
              <div className="h-1 bg-gray-200 rounded mt-2">
                <div
                  className="h-full bg-gradient-to-r from-green-600 to-red-600 rounded"
                  style={{ width: strength * 100 + '%' }}
                />
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded p-3">
              <div className="text-xs font-bold uppercase text-gray-600 mb-2">State Space</div>
              <div className="text-xl font-mono font-bold text-red-600">{Math.floor(gamma).toLocaleString()}</div>
              <div className="text-xs text-gray-500 mt-2">Γ(n,d) = d(1+d)^(n-1)</div>
            </div>
          </div>

          {/* Posterior rankings */}
          <div className="flex-1 bg-white border border-gray-200 rounded p-3">
            <h3 className="text-xs font-bold uppercase text-gray-600 mb-3 pb-2 border-b">Posterior Ranking</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {sorted.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs">
                  <div className="w-6 text-gray-500 font-bold">{idx + 1}.</div>
                  <div className="w-20">{item.region}</div>
                  <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-red-600 to-red-700"
                      style={{ width: item.prob * 100 + '%' }}
                    />
                  </div>
                  <div className="w-12 text-right font-mono font-bold">{(item.prob * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
