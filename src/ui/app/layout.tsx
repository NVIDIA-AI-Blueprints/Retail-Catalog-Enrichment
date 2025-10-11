import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "Catalog Enrichment",
  description: "AI-powered product catalog enrichment system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="nv-dark" style={{ backgroundColor: '#0c0c0c' }}>
      <head>
        {/* Load model-viewer library */}
        <Script 
          type="module" 
          src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"
          strategy="beforeInteractive"
        />
      </head>
      <body className="text-primary min-h-screen" style={{ backgroundColor: 'var(--background-color-surface-base)' }}>
        {children}
      </body>
    </html>
  );
}







