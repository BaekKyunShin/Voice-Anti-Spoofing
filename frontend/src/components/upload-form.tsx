'use client';

import { useEffect, useRef, useState } from 'react';

type ModelKey = 'gru' | 'lcnn' | 'crnn' | 'xlsr_aasist';

type ModelResult = {
  bonafide_prob: number; // ASVspoof CM score, 높을수록 진짜
  prediction: 'bonafide' | 'spoof';
  inference_ms: number;
};

type Consensus = {
  prediction: 'bonafide' | 'spoof';
  agreement: number; // 0.5, 0.75, 1.0 ...
};

type PredictionResult = {
  filename?: string;
  sample_rate?: number;
  duration_sec?: number;
  total_inference_ms?: number;
  models: Record<ModelKey, ModelResult>;
  consensus: Consensus;
};

const MODEL_ORDER: ModelKey[] = ['gru', 'lcnn', 'crnn', 'xlsr_aasist'];
const MODEL_LABEL: Record<ModelKey, string> = {
  gru: 'GRU',
  lcnn: 'LCNN',
  crnn: 'CRNN',
  xlsr_aasist: 'XLS-R + AASIST',
};

const ANALYZING_STEPS = [
  'Decoding WAV…',
  'Computing mel-spectrogram…',
  'Running GRU…',
  'Running LCNN…',
  'Running CRNN…',
  'Running XLS-R+AASIST…',
];
const STEP_INTERVAL_MS = 850;
const COLD_START_HINT_MS = 8000;

type ApiHealth = 'unknown' | 'online' | 'offline';
type Stage = 'IDLE' | 'ANALYZING' | 'RESULT';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const CARD_W = 700;
const BODY_H = 408;
const GLITCH_MS = 420;

type WaveTone = 'analyzing' | 'fake' | 'real';

/* ────────────────────────────────────────────────────────────
   ANALYZING / RESULT 화면 하단의 캔버스 파동
   ──────────────────────────────────────────────────────────── */
function Wave({ tone }: { tone: WaveTone }) {
  const cv = useRef<HTMLCanvasElement>(null);
  const raf = useRef<number | null>(null);
  const t = useRef(0);

  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const W = c.width;
    const H = c.height;
    const isFake = tone === 'fake';

    const palette = isFake
      ? [
          { a: 52, f: 0.011, sp: 0.85, c: '#FF2222', lw: 2.8, op: 0.9, fill: true },
          { a: 34, f: 0.019, sp: 1.25, c: '#FF0000', lw: 2.2, op: 0.7, fill: true },
          { a: 18, f: 0.03, sp: 0.6, c: '#FF5555', lw: 1.5, op: 0.45, fill: false },
        ]
      : [
          { a: 52, f: 0.011, sp: 0.85, c: '#22D3EE', lw: 2.8, op: 0.9, fill: true },
          { a: 34, f: 0.019, sp: 1.25, c: '#8B5CF6', lw: 2.2, op: 0.7, fill: true },
          { a: 18, f: 0.03, sp: 0.6, c: '#22D3EE', lw: 1.5, op: 0.45, fill: false },
        ];

    const gridStroke = isFake ? 'rgba(160,0,0,.1)' : 'rgba(34,211,238,.055)';
    const ampMul = tone === 'analyzing' ? 1.3 : isFake ? 1.7 : 1.4;

    const go = () => {
      t.current += 0.018;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = gridStroke;
      ctx.lineWidth = 0.5;
      for (let i = 0; i < W; i += 32) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, H);
        ctx.stroke();
      }
      for (let j = 0; j < H; j += 18) {
        ctx.beginPath();
        ctx.moveTo(0, j);
        ctx.lineTo(W, j);
        ctx.stroke();
      }

      const by = H * 0.52;
      palette.forEach((w) => {
        const pts: [number, number][] = [];
        for (let px = 0; px <= W; px += 2) {
          const py =
            by +
            Math.sin(px * w.f + t.current * w.sp) * w.a * ampMul +
            Math.sin(px * w.f * 2.2 + t.current * w.sp * 0.6) * w.a * ampMul * 0.35 +
            Math.sin(px * w.f * 0.5 + t.current * w.sp * 1.6) * w.a * ampMul * 0.18;
          pts.push([px, py]);
        }
        ctx.save();
        if (w.fill) {
          const g = ctx.createLinearGradient(0, by - w.a * ampMul * 1.3, 0, H);
          g.addColorStop(0, w.c + '60');
          g.addColorStop(0.6, w.c + '1A');
          g.addColorStop(1, w.c + '00');
          ctx.beginPath();
          ctx.moveTo(0, H);
          pts.forEach(([px, py]) => ctx.lineTo(px, py));
          ctx.lineTo(W, H);
          ctx.closePath();
          ctx.fillStyle = g;
          ctx.globalAlpha = w.op * 0.85;
          ctx.fill();
        }
        ctx.shadowBlur = 22;
        ctx.shadowColor = w.c;
        ctx.strokeStyle = w.c;
        ctx.lineWidth = w.lw;
        ctx.globalAlpha = w.op;
        ctx.beginPath();
        pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
        ctx.stroke();
        ctx.restore();
      });
      raf.current = requestAnimationFrame(go);
    };
    raf.current = requestAnimationFrame(go);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [tone]);

  return (
    <canvas
      ref={cv}
      width={CARD_W}
      height={192}
      style={{ width: '100%', height: '192px', display: 'block' }}
    />
  );
}

/* ────────────────────────────────────────────────────────────
   IDLE 화면 중앙의 흐르는 시안 파동
   ──────────────────────────────────────────────────────────── */
function IdleWave() {
  const cv = useRef<HTMLCanvasElement>(null);
  const raf = useRef<number | null>(null);
  const t = useRef(0);

  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const W = c.width;
    const H = c.height;

    const go = () => {
      t.current += 0.014;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(34,211,238,.04)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < W; i += 28) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, H);
        ctx.stroke();
      }
      for (let j = 0; j < H; j += 16) {
        ctx.beginPath();
        ctx.moveTo(0, j);
        ctx.lineTo(W, j);
        ctx.stroke();
      }
      const by = H * 0.5;
      [
        { a: 22, f: 0.016, sp: 0.6, c: '#22D3EE', lw: 2, op: 0.7, fill: true },
        { a: 14, f: 0.026, sp: 0.95, c: '#7C3AED', lw: 1.5, op: 0.5, fill: true },
        { a: 9, f: 0.038, sp: 0.45, c: '#22D3EE', lw: 1, op: 0.3, fill: false },
      ].forEach((w) => {
        const pts: [number, number][] = [];
        for (let px = 0; px <= W; px += 2) {
          const py =
            by +
            Math.sin(px * w.f + t.current * w.sp) * w.a +
            Math.sin(px * w.f * 2.1 + t.current * w.sp * 0.7) * w.a * 0.3;
          pts.push([px, py]);
        }
        ctx.save();
        if (w.fill) {
          const g = ctx.createLinearGradient(0, by - w.a * 1.2, 0, H);
          g.addColorStop(0, w.c + '40');
          g.addColorStop(0.7, w.c + '10');
          g.addColorStop(1, w.c + '00');
          ctx.beginPath();
          ctx.moveTo(0, H);
          pts.forEach(([px, py]) => ctx.lineTo(px, py));
          ctx.lineTo(W, H);
          ctx.closePath();
          ctx.fillStyle = g;
          ctx.globalAlpha = w.op * 0.7;
          ctx.fill();
        }
        ctx.shadowBlur = 12;
        ctx.shadowColor = w.c;
        ctx.strokeStyle = w.c;
        ctx.lineWidth = w.lw;
        ctx.globalAlpha = w.op;
        ctx.beginPath();
        pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
        ctx.stroke();
        ctx.restore();
      });
      raf.current = requestAnimationFrame(go);
    };
    raf.current = requestAnimationFrame(go);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, []);

  return (
    <canvas
      ref={cv}
      width={CARD_W}
      height={130}
      style={{ width: '100%', height: '130px', display: 'block' }}
    />
  );
}

/* ── 모서리 장식 ── */
function Corners({ color = 'rgba(34,211,238,.4)' }: { color?: string }) {
  const s = `1px solid ${color}`;
  const base: React.CSSProperties = { position: 'absolute', width: 13, height: 13 };
  return (
    <>
      <div className="anim-cb" aria-hidden="true" style={{ ...base, top: 10, left: 10, borderTop: s, borderLeft: s }} />
      <div className="anim-cb" aria-hidden="true" style={{ ...base, top: 10, right: 10, borderTop: s, borderRight: s }} />
      <div className="anim-cb" aria-hidden="true" style={{ ...base, bottom: 10, left: 10, borderBottom: s, borderLeft: s }} />
      <div className="anim-cb" aria-hidden="true" style={{ ...base, bottom: 10, right: 10, borderBottom: s, borderRight: s }} />
    </>
  );
}

/* ────────────────────────────────────────────────────────────
   메인
   ──────────────────────────────────────────────────────────── */
export function UploadForm() {
  const [stage, setStage] = useState<Stage>('IDLE');
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [glitch, setGlitch] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiHealth, setApiHealth] = useState<ApiHealth>('unknown');
  const [analyzingStep, setAnalyzingStep] = useState(0);
  const [showColdStartHint, setShowColdStartHint] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ANALYZING 단계 텍스트 순환 + 콜드스타트 힌트 */
  useEffect(() => {
    if (stage !== 'ANALYZING') {
      setAnalyzingStep(0);
      setShowColdStartHint(false);
      return;
    }
    setAnalyzingStep(0);
    setShowColdStartHint(false);
    const stepTimer = setInterval(() => {
      setAnalyzingStep((s) => Math.min(s + 1, ANALYZING_STEPS.length - 1));
    }, STEP_INTERVAL_MS);
    const hintTimer = setTimeout(
      () => setShowColdStartHint(true),
      COLD_START_HINT_MS,
    );
    return () => {
      clearInterval(stepTimer);
      clearTimeout(hintTimer);
    };
  }, [stage]);

  /* API 헬스 폴링 */
  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { cache: 'no-store' });
        if (!cancelled) setApiHealth(res.ok ? 'online' : 'offline');
      } catch {
        if (!cancelled) setApiHealth('offline');
      }
    };
    ping();
    const id = setInterval(ping, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const handleFile = (selected: File | null) => {
    setError(null);
    if (!selected) {
      setFile(null);
      return;
    }
    if (!selected.name.toLowerCase().endsWith('.wav')) {
      setError('WAV 파일만 업로드 가능합니다. 다른 파일을 선택해 주세요.');
      return;
    }
    setFile(selected);
  };

  const startScan = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setStage('ANALYZING');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `요청 실패 (${res.status})`);
      }
      const data: PredictionResult = await res.json();
      // 글리치 → RESULT (consensus spoof면 글리치, bonafide는 부드럽게)
      if (data.consensus.prediction === 'spoof') {
        setGlitch(true);
        setTimeout(() => {
          setGlitch(false);
          setResult(data);
          setStage('RESULT');
        }, GLITCH_MS);
      } else {
        setResult(data);
        setStage('RESULT');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '알 수 없는 오류');
      setStage('IDLE');
    }
  };

  const reset = () => {
    setStage('IDLE');
    setFile(null);
    setResult(null);
    setError(null);
    setGlitch(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  const isFake = result?.consensus.prediction === 'spoof';
  const agreeCount = result ? Math.round(result.consensus.agreement * 4) : 0;

  /* 색·테마 ─ FAKE = 빨강, REAL = 시안 (v3 원본과 동일한 값) */
  const themeRgb = isFake ? '255,0,0' : '34,211,238';
  const themeBorder = isFake ? 'rgba(255,45,45,.5)' : 'rgba(34,211,238,.5)';
  const idleBorder = 'rgba(34,211,238,.22)';
  const cardBorder = stage === 'RESULT' ? themeBorder : idleBorder;
  const cardShadow = `0 0 80px rgba(${stage === 'RESULT' ? themeRgb : '34,211,238'},.18),0 28px 80px rgba(0,0,0,.95)`;

  return (
    <div
      style={{
        width: '100%',
        maxWidth: CARD_W + 'px',
        fontFamily: "'Courier New',Courier,monospace",
        WebkitFontSmoothing: 'subpixel-antialiased',
        MozOsxFontSmoothing: 'auto',
        zoom: 1.2,
      }}
    >
      <div
        style={{
          background: 'linear-gradient(155deg,rgba(13,19,46,.97) 0%,rgba(6,9,22,.99) 100%)',
          border: `1px solid ${cardBorder}`,
          borderRadius: 16,
          overflow: 'hidden',
          boxShadow: cardShadow,
          transition: 'border-color .5s,box-shadow .5s',
          position: 'relative',
        }}
      >
        {/* 글리치 오버레이 (FAKE 진입 시) */}
        {glitch && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 60, pointerEvents: 'none' }}>
            <div
              className="anim-glitch-r"
              style={{ position: 'absolute', inset: 0, background: 'rgba(255,0,0,.32)' }}
            />
            <div
              className="anim-glitch-b"
              style={{ position: 'absolute', inset: 0, background: 'rgba(0,80,255,.18)' }}
            />
            <div
              style={{
                position: 'absolute',
                inset: 0,
                backgroundImage:
                  'repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,0,0,.1) 2px,rgba(255,0,0,.1) 3px)',
              }}
            />
          </div>
        )}

        {/* scanline (RESULT 단계, FAKE만) */}
        {stage === 'RESULT' && isFake && (
          <div
            className="anim-scan"
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              height: '2px',
              background: 'rgba(255,50,50,.14)',
              pointerEvents: 'none',
              zIndex: 4,
            }}
          />
        )}

        {/* 헤더 */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '13px 20px',
            borderBottom: `1px solid ${cardBorder}`,
            background: `rgba(${stage === 'RESULT' ? themeRgb : '34,211,238'},.04)`,
            transition: 'background-color .5s, border-color .5s',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: '50%',
                background: `rgba(${stage === 'RESULT' ? themeRgb : '34,211,238'},.15)`,
                border: `1px solid rgba(${stage === 'RESULT' ? themeRgb : '34,211,238'},.5)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 13,
                transition: 'background-color .5s, border-color .5s',
              }}
              aria-hidden="true"
            >
              🎙️
            </div>
            <span
              style={{
                fontSize: 12,
                fontWeight: 500,
                letterSpacing: '.07em',
                color: 'rgba(255,255,255,.82)',
              }}
            >
              Voice Authenticity Detector
            </span>
          </div>
          {stage !== 'IDLE' && (
            <button
              onClick={reset}
              className="reset-btn"
              style={{
                background: 'transparent',
                border: 'none',
                color: 'rgba(255,255,255,.45)',
                cursor: 'pointer',
                fontSize: 22,
                lineHeight: 1,
                padding: '0 6px',
                borderRadius: 4,
                transition: 'color .15s, background-color .15s',
              }}
              aria-label="결과 닫고 처음으로"
            >
              ×
            </button>
          )}
        </div>

        {/* ════ IDLE ════ */}
        {stage === 'IDLE' && (
          <div
            style={{
              height: BODY_H + 'px',
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Corners />

            {/* 세션 ID */}
            <div
              style={{
                position: 'absolute',
                top: 18,
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '8px',
                letterSpacing: '.38em',
                color: 'rgba(34,211,238,.35)',
                whiteSpace: 'nowrap',
                fontWeight: 300,
                zIndex: 2,
              }}
            >
              ENCRYPTED SESSION // UNIT 01
            </div>

            {/* 타이틀 */}
            <div
              style={{
                flex: '0 0 auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                paddingTop: 42,
                paddingBottom: 0,
                zIndex: 2,
              }}
            >
              <div
                style={{
                  fontSize: '28px',
                  fontWeight: 900,
                  letterSpacing: '.2em',
                  color: 'rgba(255,255,255,.92)',
                  whiteSpace: 'nowrap',
                  lineHeight: 1,
                  textShadow: '0 0 40px rgba(34,211,238,.3)',
                }}
              >
                VOICE AUTH PROTOCOL
              </div>
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ height: '1px', width: 44, background: 'rgba(34,211,238,.28)' }} />
                <span
                  style={{
                    fontSize: '7px',
                    letterSpacing: '.32em',
                    color: 'rgba(34,211,238,.38)',
                  }}
                >
                  BIOMETRIC ANALYSIS ENGINE
                </span>
                <div style={{ height: '1px', width: 44, background: 'rgba(34,211,238,.28)' }} />
              </div>
            </div>

            {/* 흐르는 파동 (배경 레이어) — v3 원본과 동일하게 가운데 비움. 카드 전체가 drop zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                handleFile(e.dataTransfer.files?.[0] ?? null);
              }}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
                overflow: 'hidden',
                outline: isDragging ? '1px dashed rgba(34,211,238,.5)' : 'none',
                outlineOffset: -8,
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  bottom: 0,
                  width: 60,
                  zIndex: 3,
                  background: 'linear-gradient(to right,rgba(8,12,28,.97),transparent)',
                  pointerEvents: 'none',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  right: 0,
                  bottom: 0,
                  width: 60,
                  zIndex: 3,
                  background: 'linear-gradient(to left,rgba(8,12,28,.97),transparent)',
                  pointerEvents: 'none',
                }}
              />
              <IdleWave />
            </div>

            <input
              ref={inputRef}
              id="audio-input"
              type="file"
              accept=".wav,audio/wav,audio/x-wav"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />

            {/* 액션 영역: 파일명(있을 때) + START 버튼 */}
            <div
              style={{
                flex: '0 0 auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 8,
                paddingBottom: 38,
                zIndex: 2,
              }}
            >
              {error && (
                <div
                  role="alert"
                  style={{
                    fontSize: '9px',
                    letterSpacing: '.12em',
                    color: '#FF6666',
                    background: 'rgba(255,0,0,.06)',
                    border: '1px solid rgba(255,80,80,.3)',
                    padding: '4px 12px',
                    borderRadius: 3,
                  }}
                >
                  {error}
                </div>
              )}
              {file && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    fontSize: '8.5px',
                    letterSpacing: '.16em',
                    color: 'rgba(34,211,238,.55)',
                  }}
                >
                  <span style={{ color: 'rgba(34,211,238,.4)' }}>FILE //</span>
                  <span
                    style={{
                      color: 'rgba(255,255,255,.7)',
                      maxWidth: 280,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={file.name}
                  >
                    {file.name}
                  </span>
                  <span style={{ color: 'rgba(255,255,255,.35)', fontVariantNumeric: 'tabular-nums' }}>
                    {`${(file.size / 1024).toFixed(1)} KB`}
                  </span>
                  <button
                    onClick={() => handleFile(null)}
                    className="dismiss-btn"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'rgba(255,255,255,.45)',
                      cursor: 'pointer',
                      fontSize: 14,
                      lineHeight: 1,
                      padding: '0 4px',
                      borderRadius: 3,
                      fontFamily: 'inherit',
                      transition: 'color .15s',
                    }}
                    aria-label="선택한 파일 제거"
                  >
                    ×
                  </button>
                </div>
              )}
              <button
                onClick={() => {
                  if (apiHealth === 'offline') {
                    setError('백엔드 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.');
                    return;
                  }
                  if (apiHealth === 'unknown') return; // 워밍업 중엔 무시
                  if (file) startScan();
                  else inputRef.current?.click();
                }}
                disabled={apiHealth === 'unknown'}
                className="start-btn"
                style={{
                  background: 'rgba(34,211,238,.05)',
                  border: '1px solid rgba(34,211,238,.55)',
                  color: '#67E8F9',
                  cursor: apiHealth === 'unknown' ? 'wait' : 'pointer',
                  padding: '12px 40px',
                  fontSize: '12px',
                  fontWeight: 700,
                  letterSpacing: '.24em',
                  borderRadius: 3,
                  boxShadow:
                    '0 0 22px rgba(34,211,238,.22), inset 0 0 18px rgba(34,211,238,.04)',
                  textShadow: '0 0 8px rgba(34,211,238,.6)',
                  transition: 'background-color .2s, box-shadow .2s, color .2s',
                  fontFamily: 'inherit',
                  touchAction: 'manipulation',
                  opacity: apiHealth === 'unknown' ? 0.7 : 1,
                }}
              >
                {apiHealth === 'unknown'
                  ? '[ WARMING UP… ]'
                  : file
                    ? '[ START BIO-SCAN ]'
                    : '[ SELECT .WAV TO SCAN ]'}
              </button>
              {apiHealth === 'offline' && (
                <div
                  style={{
                    fontSize: '8px',
                    letterSpacing: '.18em',
                    color: 'rgba(255,80,80,.55)',
                    marginTop: 2,
                  }}
                >
                  BACKEND OFFLINE — 서비스 연결 실패
                </div>
              )}
            </div>

            {/* 우측 하단 상태 (v3 원본 톤) */}
            <div
              className="anim-bk2"
              style={{
                position: 'absolute',
                bottom: 16,
                right: 18,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: '7.5px',
                letterSpacing: '.16em',
                color: 'rgba(34,211,238,.32)',
                zIndex: 2,
              }}
            >
              {apiHealth === 'offline' && (
                <span
                  style={{
                    width: 5,
                    height: 5,
                    borderRadius: '50%',
                    background: '#FF4444',
                    boxShadow: '0 0 6px rgba(255,80,80,.7)',
                  }}
                  aria-label="backend offline"
                />
              )}
              SYSTEM STATUS: WAITING FOR INPUT…
            </div>
          </div>
        )}

        {/* ════ ANALYZING ════ */}
        {stage === 'ANALYZING' && (
          <div
            style={{
              height: BODY_H + 'px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 26,
                padding: '0 20px',
              }}
            >
              <div
                style={{
                  position: 'relative',
                  width: 144,
                  height: 144,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <div
                  className="anim-r1"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '50%',
                    border: '1px solid rgba(34,211,238,.35)',
                  }}
                />
                <div
                  className="anim-r2"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '50%',
                    border: '1px solid rgba(34,211,238,.25)',
                  }}
                />
                <div
                  className="anim-r3"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '50%',
                    border: '1px solid rgba(34,211,238,.15)',
                  }}
                />
                <div
                  className="anim-gp"
                  style={{
                    width: 98,
                    height: 98,
                    borderRadius: '50%',
                    border: '1px solid rgba(34,211,238,.75)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(34,211,238,.05)',
                    fontSize: 34,
                  }}
                  aria-hidden="true"
                >
                  🔍
                </div>
              </div>
              <div
                aria-live="polite"
                style={{
                  fontSize: 11,
                  letterSpacing: '.14em',
                  color: 'rgba(34,211,238,.7)',
                  minHeight: 16,
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {ANALYZING_STEPS[analyzingStep]}
              </div>
              {showColdStartHint && (
                <div
                  style={{
                    marginTop: -10,
                    fontSize: '8.5px',
                    letterSpacing: '.16em',
                    color: 'rgba(255,255,255,.42)',
                    textAlign: 'center',
                    maxWidth: 340,
                    lineHeight: 1.6,
                  }}
                >
                  First request can take ~30s on cold start.
                  <br />
                  Models warm up after the first scan…
                </div>
              )}
            </div>
            <Wave tone="analyzing" />
          </div>
        )}

        {/* ════ RESULT (4-모델 consensus + 카드 그리드) ════ */}
        {stage === 'RESULT' && result && (
          <div
            className="anim-fi"
            role="status"
            aria-live="polite"
            style={{
              minHeight: BODY_H + 'px',
              display: 'flex',
              flexDirection: 'column',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {/* 상단: consensus 배지 + 합의도 % */}
            <div
              style={{
                padding: '22px 26px 14px',
                textAlign: 'center',
              }}
            >
              <span
                className={isFake ? 'anim-bk' : undefined}
                style={{
                  display: 'inline-block',
                  background: isFake ? 'rgba(255,0,0,.14)' : 'rgba(34,211,238,.12)',
                  border: `1px solid ${
                    isFake ? 'rgba(255,50,50,.7)' : 'rgba(34,211,238,.65)'
                  }`,
                  color: isFake ? '#FF4444' : '#22D3EE',
                  fontSize: '11px',
                  fontWeight: 700,
                  padding: '4px 18px',
                  borderRadius: 3,
                  letterSpacing: '.2em',
                  boxShadow: isFake
                    ? '0 0 16px rgba(255,0,0,.4)'
                    : '0 0 16px rgba(34,211,238,.3)',
                }}
              >
                {agreeCount}/4 MODELS · {isFake ? 'FAKE' : 'AUTHENTIC'}
              </span>
              <div
                className="anim-ci"
                style={{
                  marginTop: 10,
                  fontSize: 'clamp(64px,10vw,92px)',
                  fontWeight: 900,
                  lineHeight: 1,
                  color: isFake ? '#FF6262' : '#22D3EE',
                  letterSpacing: '.04em',
                  textShadow: isFake
                    ? '0 0 50px rgba(255,80,80,.65)'
                    : '0 0 50px rgba(34,211,238,.6)',
                }}
              >
                {isFake ? 'FAKE' : 'AUTHENTIC'}
              </div>
              <div
                style={{
                  marginTop: 6,
                  fontSize: 11,
                  color: 'rgba(255,255,255,.42)',
                  letterSpacing: '.06em',
                }}
              >
                Final verdict · {agreeCount}/4 models agreed
              </div>
            </div>

            {/* 중단: 4-모델 카드 2×2 그리드 */}
            <div
              style={{
                padding: '0 20px 14px',
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 10,
              }}
            >
              {MODEL_ORDER.map((key, i) => {
                const m = result.models[key];
                const mIsFake = m.prediction === 'spoof';
                const mRgb = mIsFake ? '255,68,68' : '34,211,238';
                const bonafidePct = m.bonafide_prob * 100;
                return (
                  <div
                    key={key}
                    className="anim-card-in"
                    style={{
                      border: `1px solid rgba(${mRgb},.5)`,
                      background: `linear-gradient(135deg, rgba(${mRgb},.1), rgba(${mRgb},.02))`,
                      borderRadius: 6,
                      padding: '12px 14px',
                      position: 'relative',
                      overflow: 'hidden',
                      boxShadow: `0 0 16px rgba(${mRgb},.12)`,
                      animationDelay: `${i * 120}ms`,
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 8,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 11,
                          fontWeight: 700,
                          letterSpacing: '.14em',
                          color: 'rgba(255,255,255,.92)',
                        }}
                      >
                        {MODEL_LABEL[key]}
                      </span>
                      <span
                        style={{
                          fontSize: 9.5,
                          fontWeight: 700,
                          letterSpacing: '.18em',
                          padding: '2px 8px',
                          border: `1px solid rgba(${mRgb},.8)`,
                          color: mIsFake ? '#FF7676' : '#7DEAF7',
                          background: `rgba(${mRgb},.18)`,
                          borderRadius: 2,
                        }}
                      >
                        {mIsFake ? 'FAKE' : 'AUTHENTIC'}
                      </span>
                    </div>

                    {/* 메인 데이터 — bonafide_prob (ASVspoof CM score)를 크게 강조 */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'baseline',
                        justifyContent: 'space-between',
                        marginBottom: 6,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                        <span
                          style={{
                            fontSize: 9,
                            fontWeight: 600,
                            letterSpacing: '.18em',
                            color: 'rgba(255,255,255,.45)',
                            textTransform: 'uppercase',
                          }}
                        >
                          bonafide
                        </span>
                        <span
                          style={{
                            fontSize: 22,
                            fontWeight: 800,
                            lineHeight: 1,
                            color: mIsFake ? '#FF8888' : '#9CEEF7',
                            letterSpacing: '-.02em',
                            textShadow: `0 0 12px rgba(${mRgb},.45)`,
                          }}
                        >
                          {bonafidePct.toFixed(1)}
                          <span style={{ fontSize: 13, marginLeft: 2, opacity: 0.7 }}>%</span>
                        </span>
                      </div>
                      <span
                        style={{
                          fontSize: 10,
                          color: 'rgba(255,255,255,.55)',
                          letterSpacing: '.04em',
                          fontWeight: 500,
                          fontVariantNumeric: 'tabular-nums',
                        }}
                      >
                        {`${m.inference_ms.toFixed(0)} ms`}
                      </span>
                    </div>

                    {/* 확률 바 (bonafide_prob 기준 — 꽉 찰수록 진짜 자신감) */}
                    <div
                      style={{
                        position: 'relative',
                        height: 6,
                        background: 'rgba(255,255,255,.07)',
                        borderRadius: 3,
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${bonafidePct}%`,
                          background: `linear-gradient(90deg, rgba(${mRgb},.55), rgba(${mRgb},1))`,
                          boxShadow: `0 0 10px rgba(${mRgb},.65)`,
                          transition: 'width .6s ease-out',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 하단: 파일정보 + 총 추론 시간 */}
            <div
              style={{
                flexShrink: 0,
                padding: '12px 20px 16px',
                borderTop: `1px solid rgba(${themeRgb},.22)`,
                background: `rgba(${themeRgb},.05)`,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: 16,
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  color: 'rgba(255,255,255,.5)',
                  letterSpacing: '.06em',
                  fontWeight: 600,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                  minWidth: 0,
                }}
                title={result.filename ?? file?.name ?? 'unknown.wav'}
              >
                <span style={{ color: 'rgba(255,255,255,.32)', marginRight: 6 }}>FILE //</span>
                <span style={{ color: 'rgba(255,255,255,.92)', fontWeight: 600 }}>
                  {result.filename ?? file?.name ?? 'unknown.wav'}
                </span>
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: 'rgba(255,255,255,.55)',
                  letterSpacing: '.06em',
                  fontWeight: 500,
                  whiteSpace: 'nowrap',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {result.total_inference_ms !== undefined
                  ? `${result.total_inference_ms.toFixed(0)} ms`
                  : 'Analysed'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
