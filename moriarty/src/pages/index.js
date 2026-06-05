import React, { Suspense } from "react";
import Link from "next/link";
import Head from "next/head";
import { Play } from "lucide-react";
import dynamic from "next/dynamic";

const GlobeViewer = dynamic(() => import("@/components/GlobeViewer"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="text-center text-white">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4" />
        <p>Loading paleogeographic globe...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Sighthound — Phase-Locked Positioning</title>
        <meta
          name="description"
          content="Phase-Locked Positioning & Atmospheric Triangulation Framework"
        />
      </Head>

      <div className="h-screen w-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-hidden">
        {/* 3D Globe Viewer */}
        <Suspense
          fallback={
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center text-white">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4" />
                <p>Loading...</p>
              </div>
            </div>
          }
        >
          <GlobeViewer />
        </Suspense>

        {/* Navigation */}
        <nav className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-8 py-6">
          <div>
            <h1 className="text-2xl font-bold text-white">Sighthound</h1>
            <p className="text-xs text-slate-400">Phase-Locked Positioning</p>
          </div>
          <Link
            href="/sandbox"
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-semibold transition-all text-white shadow-lg hover:shadow-xl hover:scale-105"
          >
            <Play className="w-5 h-5" />
            Sandbox
          </Link>
        </nav>
      </div>
    </>
  );
}
