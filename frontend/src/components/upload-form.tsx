'use client';

import { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Loader2,
  ShieldCheck,
  ShieldAlert,
  X,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Waveform } from '@/components/waveform';
import { AnimatedCounter } from '@/components/animated-counter';
import { cn } from '@/lib/utils';

type PredictionResult = {
  real_prob: number;
  fake_prob: number;
  prediction: 'real' | 'fake';
  inference_ms?: number;
  filename?: string;
  duration_sec?: number;
  sample_rate?: number;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type ApiHealth = 'unknown' | 'online' | 'offline';

export function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [apiHealth, setApiHealth] = useState<ApiHealth>('unknown');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, {
          cache: 'no-store',
        });
        if (!cancelled) setApiHealth(res.ok ? 'online' : 'offline');
      } catch {
        if (!cancelled) setApiHealth('offline');
      }
    };
    ping();
    const interval = setInterval(ping, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const handleFile = (selected: File | null) => {
    setError(null);
    setResult(null);
    if (!selected) {
      setFile(null);
      return;
    }
    if (!selected.name.toLowerCase().endsWith('.wav')) {
      setError('WAV 파일만 업로드 가능합니다.');
      return;
    }
    setFile(selected);
  };

  const onSubmit = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `요청 실패 (${res.status})`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : '알 수 없는 오류');
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const isReal = result?.prediction === 'real';
  const realPct = result ? result.real_prob * 100 : 0;
  const fakePct = result ? result.fake_prob * 100 : 0;
  const confidence = result
    ? Math.max(result.real_prob, result.fake_prob) * 100
    : 0;

  return (
    <div className="w-full max-w-xl flex flex-col gap-3">
      {/* 상단 상태 배지 */}
      <div className="flex items-center justify-between text-xs px-1">
        <div className="flex items-center gap-2 font-mono uppercase tracking-[0.2em] text-muted-foreground">
          <Sparkles className="size-3" />
          deep learning · audio anti-spoofing
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'relative flex size-2 rounded-full',
              apiHealth === 'online' && 'bg-emerald-500',
              apiHealth === 'offline' && 'bg-rose-500',
              apiHealth === 'unknown' && 'bg-zinc-500'
            )}
          >
            {apiHealth === 'online' && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            )}
          </span>
          <span className="font-mono text-muted-foreground">
            {apiHealth === 'online'
              ? 'API connected'
              : apiHealth === 'offline'
                ? 'API offline'
                : 'connecting…'}
          </span>
        </div>
      </div>

      {/* 메인 카드 */}
      <Card
        className={cn(
          'relative bg-card/70 backdrop-blur-xl border-border/50 shadow-2xl overflow-hidden transition-shadow duration-700',
          result && isReal && 'shadow-emerald-500/20',
          result && !isReal && 'shadow-rose-500/20'
        )}
      >
        {/* 미묘한 그라디언트 보더 */}
        <div
          aria-hidden
          className={cn(
            'pointer-events-none absolute inset-0 rounded-xl opacity-40 transition-colors duration-700',
            'bg-[radial-gradient(ellipse_at_top,var(--color-foreground)/0.08,transparent_60%)]',
            result && isReal && 'opacity-60',
            result && !isReal && 'opacity-60'
          )}
        />

        <CardHeader className="relative text-center gap-2">
          <CardTitle className="text-2xl font-mono tracking-tight">
            Voice Authenticity Detector
          </CardTitle>
          <CardDescription className="text-[15px] tracking-tight">
            WAV 파일을 업로드하면 실제 음성인지 합성 음성인지 판별합니다.
          </CardDescription>
        </CardHeader>

        <CardContent className="relative flex flex-col gap-5">
          {/* 업로드 영역 */}
          {!file ? (
            <label
              htmlFor="audio-input"
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
              className={cn(
                'group relative flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/70 px-6 py-10 text-center cursor-pointer transition-all',
                'hover:border-foreground/40 hover:bg-foreground/[0.03]',
                isDragging && 'border-foreground/60 bg-foreground/[0.05] scale-[1.01]'
              )}
            >
              <div className="rounded-full bg-foreground/5 p-3 transition-transform group-hover:scale-110">
                <Upload className="size-6 text-foreground/70" />
              </div>
              <div className="flex flex-col gap-1">
                <div className="text-sm font-medium">
                  클릭 또는 파일을 드래그해서 업로드
                </div>
                <div className="text-xs font-mono text-muted-foreground">
                  .wav · max 25MB
                </div>
              </div>
            </label>
          ) : (
            <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-foreground/[0.02] p-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex items-start justify-between gap-3">
                <div className="flex flex-col min-w-0">
                  <div className="text-sm font-medium truncate">
                    {file.name}
                  </div>
                  <div className="text-xs font-mono text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </div>
                </div>
                <button
                  onClick={reset}
                  className="text-muted-foreground hover:text-foreground transition shrink-0"
                  aria-label="파일 제거"
                  disabled={isLoading}
                >
                  <X className="size-4" />
                </button>
              </div>
              <Waveform file={file} className="w-full h-14" />
            </div>
          )}

          <input
            ref={inputRef}
            id="audio-input"
            type="file"
            accept=".wav,audio/wav,audio/x-wav"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          />

          {error && (
            <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-sm text-destructive animate-in fade-in slide-in-from-top-1 duration-200">
              {error}
            </div>
          )}

          {/* 액션 버튼 */}
          <Button
            onClick={onSubmit}
            disabled={!file || isLoading || apiHealth === 'offline'}
            className="w-full text-base font-semibold tracking-tight"
            size="lg"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" /> 분석 중...
              </>
            ) : apiHealth === 'offline' ? (
              'API 서버 오프라인'
            ) : (
              '판별하기'
            )}
          </Button>

          {/* 결과 표시 */}
          {result && (
            <div
              className={cn(
                'flex flex-col gap-4 rounded-lg border bg-background/40 p-5 animate-in fade-in slide-in-from-bottom-3 duration-500',
                isReal
                  ? 'border-emerald-500/30 shadow-[inset_0_0_60px_-30px_rgb(16_185_129/0.4)]'
                  : 'border-rose-500/30 shadow-[inset_0_0_60px_-30px_rgb(244_63_94/0.4)]'
              )}
              key={result.filename ?? '' + (result.inference_ms ?? 0)}
            >
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    'flex items-center justify-center rounded-full p-2 transition-colors',
                    isReal
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-rose-500/10 text-rose-400'
                  )}
                >
                  {isReal ? (
                    <ShieldCheck className="size-6" />
                  ) : (
                    <ShieldAlert className="size-6" />
                  )}
                </div>
                <div className="flex flex-col flex-1 min-w-0">
                  <div className="text-xl font-bold tracking-tight">
                    {isReal ? '실제 음성 (Real)' : '합성 음성 (Fake)'}
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
                    <span>
                      신뢰도{' '}
                      <span className="text-foreground tabular-nums">
                        <AnimatedCounter value={confidence} decimals={1} />%
                      </span>
                    </span>
                    {result.inference_ms !== undefined && (
                      <>
                        <span className="opacity-40">·</span>
                        <span>
                          <span className="text-foreground tabular-nums">
                            {result.inference_ms.toFixed(0)}
                          </span>{' '}
                          ms
                        </span>
                      </>
                    )}
                    {result.duration_sec !== undefined && (
                      <>
                        <span className="opacity-40">·</span>
                        <span>
                          <span className="text-foreground tabular-nums">
                            {result.duration_sec.toFixed(2)}
                          </span>{' '}
                          s
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <ProbBar
                  label="실제 음성 확률"
                  pct={realPct}
                  color="emerald"
                />
                <ProbBar
                  label="합성 음성 확률"
                  pct={fakePct}
                  color="rose"
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ProbBar({
  label,
  pct,
  color,
}: {
  label: string;
  pct: number;
  color: 'emerald' | 'rose';
}) {
  const indicatorClass =
    color === 'emerald'
      ? 'bg-gradient-to-r from-emerald-500/80 to-emerald-400'
      : 'bg-gradient-to-r from-rose-500/80 to-rose-400';
  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-medium tabular-nums">
          <AnimatedCounter value={pct} decimals={1} />%
        </span>
      </div>
      <Progress value={pct} indicatorClassName={indicatorClass} />
    </div>
  );
}
