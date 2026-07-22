import React, { Suspense } from "react";
import Link from "next/link";
import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

const GlbViewer = dynamic(() => import("@/components/GLBViewer"), {
  ssr: false,
  loading: () => null,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Sighthound</title>
        <meta
          name="description"
          content="Phase-Locked Positioning & Atmospheric Triangulation Framework"
        />
      </Head>

      <main className="relative h-screen w-full overflow-hidden bg-[#050507]">
        <TransitionEffect />

        <Suspense fallback={null}>
          <GlbViewer />
        </Suspense>

        {/* minimal wordmark — delete for a fully bare scene */}
     {/* centered see-through wordmark */}
<div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
  <h1
    className="select-none text-center text-[9w] font-semibold leading-none tracking-tight md:text-[12vw]"
    style={{
      color: "transparent",
      WebkitTextStroke: "1px rgba(255,255,255,0.5)",
    }}
  >
    Sighthound
  </h1>
</div>

      </main>
    </>
  );
}