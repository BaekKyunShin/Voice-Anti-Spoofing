import { ImageResponse } from "next/og";

export const alt = "Voice Authenticity Detector — 4 models, 1 verdict";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background:
            "radial-gradient(ellipse 110% 80% at 50% 35%,#0b1135 0%,#050508 100%)",
          color: "#fff",
          fontFamily: "system-ui, -apple-system, sans-serif",
          padding: 80,
          position: "relative",
        }}
      >
        {/* 상단 라벨 */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            color: "rgba(34,211,238,.85)",
            fontSize: 22,
            letterSpacing: 12,
            fontWeight: 600,
            marginBottom: 28,
          }}
        >
          <div style={{ width: 56, height: 1, background: "rgba(34,211,238,.5)" }} />
          BIOMETRIC ANALYSIS ENGINE
          <div style={{ width: 56, height: 1, background: "rgba(34,211,238,.5)" }} />
        </div>

        {/* 메인 타이틀 */}
        <div
          style={{
            fontSize: 96,
            fontWeight: 900,
            letterSpacing: -2,
            color: "#fff",
            lineHeight: 1,
            textAlign: "center",
            textShadow: "0 0 60px rgba(34,211,238,.45)",
          }}
        >
          Voice Authenticity Detector
        </div>

        {/* 서브 카피 */}
        <div
          style={{
            marginTop: 28,
            fontSize: 38,
            color: "rgba(255,255,255,.7)",
            fontWeight: 500,
            letterSpacing: -0.5,
          }}
        >
          4 models · 1 verdict
        </div>

        {/* 모델명 4개 */}
        <div
          style={{
            marginTop: 80,
            display: "flex",
            gap: 24,
            fontSize: 26,
            fontWeight: 700,
            letterSpacing: 3,
          }}
        >
          {["GRU", "LCNN", "CRNN", "XLS-R+AASIST"].map((m) => (
            <div
              key={m}
              style={{
                display: "flex",
                padding: "12px 26px",
                border: "1px solid rgba(34,211,238,.5)",
                background: "rgba(34,211,238,.08)",
                color: "#67E8F9",
                borderRadius: 8,
              }}
            >
              {m}
            </div>
          ))}
        </div>

        {/* 우측 하단 도메인 */}
        <div
          style={{
            position: "absolute",
            bottom: 40,
            right: 60,
            display: "flex",
            fontSize: 20,
            color: "rgba(34,211,238,.55)",
            letterSpacing: 3,
            fontWeight: 600,
          }}
        >
          voice-anti-spoofing.vercel.app
        </div>
      </div>
    ),
    { ...size },
  );
}
