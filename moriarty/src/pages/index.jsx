import dynamic from 'next/dynamic';
import Head from 'next/head';

const GlobeVisualization = dynamic(() => import('@/components/GlobeVisualization'), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Phase-Locked Messaging</title>
        <meta name="description" content="Secure communication via topological position coherence" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <main style={{ margin: 0, width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <GlobeVisualization />
      </main>
    </>
  );
}
