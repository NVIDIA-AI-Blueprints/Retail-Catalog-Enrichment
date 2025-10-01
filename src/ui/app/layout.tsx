import type { Metadata } from "next";
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
      <body className="text-primary min-h-screen" style={{ backgroundColor: 'var(--background-color-surface-base)' }}>
        {children}
      </body>
    </html>
  );
}







