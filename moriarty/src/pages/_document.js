import { Html, Head, Main, NextScript } from "next/document";
import Script from "next/script";

export default function Document() {
  return (
    <Html lang="en" className="dark">
      <Head />
      <body>
        <Script id="theme-switcher" strategy="beforeInteractive">
        {`
  // Site is permanently dark: always apply the dark class, no toggle.
  document.documentElement.classList.add('dark');
  try { localStorage.setItem('theme', 'dark'); } catch (e) {}
  `}
        </Script>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
