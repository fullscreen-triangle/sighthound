"use client";

import React, { Suspense, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  Bounds,
  Center,
  useGLTF,
  useAnimations,
} from "@react-three/drei";

function Model(props) {
  const group = useRef();
  const { nodes, materials, animations } = useGLTF("/soyuz__apollo.glb");
  const { actions } = useAnimations(animations, group);

  // Play every clip embedded in the GLB
  useEffect(() => {
    const clips = Object.values(actions);
    clips.forEach((action) => action?.reset().fadeIn(0.5).play());
    return () => clips.forEach((action) => action?.fadeOut(0.5));
  }, [actions]);

  return (
    <group ref={group} {...props} dispose={null}>
      <group name="Sketchfab_Scene">
        <group name="Sketchfab_model" rotation={[-Math.PI / 2, 0, 0]}>
          <group name="Root">
            <group
              name="Sun"
              position={[-2.704, -27.456, 12.186]}
              rotation={[1.316, -0.449, -0.112]}
            >
              <group name="Sun_1" />
            </group>
            <group name="Soyuz" position={[13.02, 0, 0]} rotation={[-Math.PI, 0, 0]}>
              <mesh geometry={nodes.Soyuz_0.geometry} material={materials["Solar_bat.001"]} />
              <mesh geometry={nodes.Soyuz_1.geometry} material={materials["Solar_bat.003"]} />
              <mesh geometry={nodes.Soyuz_2.geometry} material={materials["SOUZ1.001"]} />
              <mesh geometry={nodes.Soyuz_2_1.geometry} material={materials["SOUZ1.001"]} />
              <mesh geometry={nodes.Soyuz_3.geometry} material={materials.SOUZ2} />
              <mesh geometry={nodes.Soyuz_4.geometry} material={materials.SOUZ4} />
              <mesh geometry={nodes.Soyuz_5.geometry} material={materials["Material.007"]} />
            </group>
            <group
              name="Apollo001"
              position={[-31.6, 0, 0]}
              rotation={[0.847, Math.PI / 2, 0]}
              scale={604.547}
            >
              <mesh geometry={nodes.Apollo001_0.geometry} material={materials.Apollo} />
              <mesh geometry={nodes.Apollo001_1.geometry} material={materials["Apollo.002"]} />
              <mesh geometry={nodes.Apollo001_2.geometry} material={materials["Apollo.003"]} />
              <mesh geometry={nodes.Apollo001_3.geometry} material={materials["Apollo.005"]} />
              <mesh geometry={nodes.Apollo001_4.geometry} material={materials["Apollo.006"]} />
              <group
                name="Cylinder021"
                position={[0, 0, 0.036]}
                rotation={[-1.35, -1.49, 1.809]}
                scale={0.002}
              >
                <mesh geometry={nodes.Cylinder021_0.geometry} material={materials.Metall} />
                <mesh geometry={nodes.Cylinder021_0_1.geometry} material={materials.Metall} />
                <mesh geometry={nodes.Cylinder021_0_2.geometry} material={materials.Metall} />
                <mesh geometry={nodes.Cylinder021_0_3.geometry} material={materials.Metall} />
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

useGLTF.preload("/soyuz__apollo.glb");

export default function GlbViewer() {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 0, 60], fov: 40 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#050507"]} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[20, 15, 10]} intensity={2.2} />
      <directionalLight position={[-15, -10, -10]} intensity={0.6} />

      <Suspense fallback={null}>
        <Bounds fit clip observe margin={1.25}>
          <Center>
            <Model />
          </Center>
        </Bounds>
        <Environment preset="night" />
      </Suspense>

      <OrbitControls
        makeDefault
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        minDistance={25}
        maxDistance={180}
      />
    </Canvas>
  );
}