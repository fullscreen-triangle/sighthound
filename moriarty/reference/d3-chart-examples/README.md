# D3 chart examples (reference only)

`D3ChartsModule.tsx.example` is **reference/example code**, not shippable source.
It sketches d3-based chart ideas to draw from when building real charts for the
web tool. It was never meant to compile as-is (it imports `d3`, which is not a
project dependency, and the code is illustrative).

It lives here — outside `src/` and with a `.example` extension — so that
`next build` does not type-check it (an unresolved `d3` import previously broke
the Vercel build). Use it as a pattern source; port pieces into real components
under `src/` and add the `d3` / `@types/d3` dependencies there if/when you do.
