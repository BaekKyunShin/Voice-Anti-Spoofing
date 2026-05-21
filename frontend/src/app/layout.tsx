import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const SITE_URL = "https://voice-anti-spoofing.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "Voice Authenticity Detector",
  description:
    "실제 vs 합성 음성 4-모델 판별 데모 (GRU · LCNN · CRNN · XLS-R+AASIST)",
  openGraph: {
    title: "Voice Authenticity Detector",
    description: "4 models, 1 verdict — 실시간 음성 진위 판별 데모",
    type: "website",
    url: SITE_URL,
    siteName: "Voice Authenticity Detector",
  },
  twitter: {
    card: "summary_large_image",
    title: "Voice Authenticity Detector",
    description: "4 models, 1 verdict — 실시간 음성 진위 판별 데모",
  },
};

export const viewport: Viewport = {
  themeColor: "#0b1135",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ko"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
      style={{ colorScheme: "dark" }}
      suppressHydrationWarning
    >
      <head>
        <link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.css"
        />
      </head>
      <body className="min-h-full flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          forcedTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
