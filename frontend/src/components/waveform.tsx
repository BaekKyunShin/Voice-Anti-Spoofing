'use client';

import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

type WaveformProps = {
  file: File | null;
  className?: string;
  bars?: number;
  barColor?: string;
};

export function Waveform({
  file,
  className,
  bars = 96,
  barColor = 'currentColor',
}: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPeaks(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setError(null);

    (async () => {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const Ctx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext })
            .webkitAudioContext;
        const ctx = new Ctx();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
        const channel = audioBuffer.getChannelData(0);
        const blockSize = Math.floor(channel.length / bars);
        const result: number[] = [];
        let maxPeak = 0;
        for (let i = 0; i < bars; i++) {
          const start = i * blockSize;
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(channel[start + j] ?? 0);
          }
          const value = sum / blockSize;
          maxPeak = Math.max(maxPeak, value);
          result.push(value);
        }
        const normalized =
          maxPeak > 0 ? result.map((p) => p / maxPeak) : result;
        if (!cancelled) setPeaks(normalized);
        ctx.close();
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : '디코딩 실패');
          setPeaks(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [file, bars]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const gap = 2;
    const barWidth = (cssWidth - gap * (bars - 1)) / bars;
    const centerY = cssHeight / 2;
    ctx.fillStyle = barColor;

    peaks.forEach((peak, i) => {
      const minBarHeight = 2;
      const h = Math.max(peak * cssHeight * 0.95, minBarHeight);
      const x = i * (barWidth + gap);
      const y = centerY - h / 2;
      ctx.beginPath();
      const radius = Math.min(barWidth / 2, 2);
      ctx.roundRect(x, y, barWidth, h, radius);
      ctx.fill();
    });
  }, [peaks, bars, barColor]);

  if (error) {
    return (
      <div className={cn('text-xs text-muted-foreground italic', className)}>
        파형 미리보기를 만들 수 없습니다 ({error})
      </div>
    );
  }

  if (!peaks) {
    return (
      <div
        className={cn(
          'flex items-center justify-center h-12 text-xs text-muted-foreground',
          className
        )}
      >
        <div className="flex gap-[3px]">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="w-[2px] bg-muted-foreground/40 rounded-full animate-pulse"
              style={{
                height: `${10 + Math.sin(i) * 6 + 6}px`,
                animationDelay: `${i * 50}ms`,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className={cn('w-full h-12 text-foreground/70', className)}
    />
  );
}
