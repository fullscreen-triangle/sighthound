import React from "react";
import Link from "next/link";
import { Code2, Play, BookOpen, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-slate-700">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <h1 className="text-4xl font-bold mb-2">Sighthound</h1>
          <p className="text-slate-300">Phase-Locked Positioning & Atmospheric Triangulation</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-4">Learn Phase-Locked Positioning</h2>
              <p className="text-slate-300 mb-6">
                Explore a unified framework for satellite-free positioning through topological coherence and atmospheric partition triangulation.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <BookOpen className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold mb-1">Progressive Tutorials</h3>
                    <p className="text-slate-400 text-sm">Learn from Level 1 basics to Phase-Locked coherence</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Code2 className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold mb-1">CyneScript Language</h3>
                    <p className="text-slate-400 text-sm">Declarative DSL with dimensional type checking</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Zap className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold mb-1">Real-Time Execution</h3>
                    <p className="text-slate-400 text-sm">Run scripts with your own GeoJSON/KML data</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <img
                src="https://images.unsplash.com/photo-1526374965328-7f5ae4e8b08f?w=600&h=400&fit=crop"
                alt="Globe visualization"
                className="rounded-lg w-full"
              />
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-slate-800 rounded-lg border border-slate-700 p-8 mb-16">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-3">Ready to explore?</h2>
            <p className="text-slate-300 mb-6">
              Open the interactive sandbox to run CyneScript tutorials and validate positioning algorithms with your own data.
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/sandbox"
              className="inline-flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-semibold transition-colors"
            >
              <Play className="w-5 h-5" />
              Open Sandbox
            </Link>
            <Link
              href="/docs"
              className="inline-flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 px-6 py-3 rounded-lg font-semibold transition-colors"
            >
              <BookOpen className="w-5 h-5" />
              Read Docs
            </Link>
          </div>
        </section>

        {/* Features Grid */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-8">Tutorial Levels</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                level: 1,
                title: "Load & Display",
                desc: "Understand data formats and basic file operations",
                color: "from-blue-500 to-blue-600",
              },
              {
                level: 2,
                title: "S-Entropy Computation",
                desc: "Calculate partition state from geographic coordinates",
                color: "from-green-500 to-green-600",
              },
              {
                level: 3,
                title: "Virtual Positioning",
                desc: "Derive virtual satellites and triangulate positions",
                color: "from-purple-500 to-purple-600",
              },
              {
                level: 4,
                title: "Dynamic Tracking",
                desc: "Real-time tracking with state estimation",
                color: "from-orange-500 to-orange-600",
              },
              {
                level: 5,
                title: "Phase-Locked Coherence",
                desc: "Establish coherence between multiple devices",
                color: "from-red-500 to-red-600",
              },
              {
                level: "?",
                title: "Your Experiment",
                desc: "Use the sandbox to test your own ideas",
                color: "from-slate-600 to-slate-700",
              },
            ].map((item) => (
              <div
                key={item.level}
                className={`bg-gradient-to-br ${item.color} rounded-lg p-6 text-white border border-white border-opacity-20`}
              >
                <div className="text-4xl font-bold mb-2 opacity-80">Level {item.level}</div>
                <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                <p className="text-white text-opacity-90">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Code Example */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6">CyneScript Example</h2>
          <div className="bg-slate-900 rounded-lg border border-slate-700 p-6 overflow-x-auto">
            <pre className="font-mono text-sm text-slate-300">
              {`# Load your trajectory data
load geojson from "trajectory.geojson"

# Initialize atmosphere model
atmosphere initialize model="standard"

# Create virtual satellites
satellite derive from Earth partition
satellite create count=1000

# Run positioning for each point
for each point in trajectory:
    atmosphere compute at point.lat point.lon point.elevation
    entropy get Sk St Se
    position resolve from entropy
    results store point.id position
end

# Display and validate results
show results
validate accuracy target=1cm`}
            </pre>
          </div>
        </section>

        {/* Features */}
        <section>
          <h2 className="text-2xl font-bold mb-8">Why Phase-Locked Positioning?</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-3 text-blue-400">No Satellites Required</h3>
              <p className="text-slate-300">
                Derive virtual satellites from Earth's partition structure. No infrastructure cost, no orbiting hardware, pure mathematics.
              </p>
            </div>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-3 text-green-400">1cm Accuracy</h3>
              <p className="text-slate-300">
                192× better than traditional GPS. Works indoors, underwater, through obstacles. Accuracy independent of environment.
              </p>
            </div>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-3 text-purple-400">Instantaneous Finality</h3>
              <p className="text-slate-300">
                No consensus needed. State changes are immediately observable to coherent parties. <1ms settlement time.
              </p>
            </div>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-3 text-orange-400">Zero Transmission</h3>
              <p className="text-slate-300">
                Position determined through observation, not data exchange. Perfect privacy. No broadcasts, no records.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700 mt-16">
        <div className="max-w-6xl mx-auto px-6 py-8 text-center text-slate-400 text-sm">
          <p>Sighthound — Phase-Locked Positioning Framework</p>
        </div>
      </footer>
    </div>
  );
}
