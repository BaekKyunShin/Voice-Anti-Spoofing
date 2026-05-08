'use client';

import { useState, useRef } from 'react';
import { Upload, AudioLines, Loader2, ShieldCheck, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

type PredictionResult = {
  real_prob: number;
  fake_prob: number;
  prediction: 'real' | 'fake';
  inference_ms?: number;
  filename?: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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
  const realPct = result ? Math.round(result.real_prob * 100) : 0;
  const fakePct = result ? Math.round(result.fake_prob * 100) : 0;

  return (
    <Card className="w-full max-w-xl bg-card/80 backdrop-blur-xl border-border/60 shadow-2xl">
      <CardHeader className="text-center gap-2">
        <CardTitle className="text-2xl font-mono tracking-tight">
          Voice Authenticity Detector
        </CardTitle>
        <CardDescription className="text-[15px] font-medium tracking-tight">
          WAV 파일을 업로드하면 실제 음성인지 합성 음성인지 판별합니다.
        </CardDescription>
      </CardHeader>

      <CardContent className="flex flex-col gap-5">
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
            const dropped = e.dataTransfer.files?.[0] ?? null;
            handleFile(dropped);
          }}
          className={cn(
            'flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-border/70 px-6 py-10 text-center transition cursor-pointer hover:border-primary/60 hover:bg-accent/40',
            isDragging && 'border-primary bg-accent/60'
          )}
        >
          {file ? (
            <>
              <AudioLines className="size-8 text-primary" />
              <div className="text-sm font-medium">{file.name}</div>
              <div className="text-xs text-muted-foreground">
                {(file.size / 1024).toFixed(1)} KB · 클릭해서 다른 파일 선택
              </div>
            </>
          ) : (
            <>
              <Upload className="size-8 text-muted-foreground" />
              <div className="text-sm">
                <span className="font-medium">클릭</span> 또는 파일을 드래그해서
                업로드
              </div>
              <div className="text-xs text-muted-foreground">.wav 파일만 지원</div>
            </>
          )}
          <input
            ref={inputRef}
            id="audio-input"
            type="file"
            accept=".wav,audio/wav,audio/x-wav"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          />
        </label>

        {error && (
          <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="flex gap-2">
          <Button
            onClick={onSubmit}
            disabled={!file || isLoading}
            className="flex-1 text-base font-semibold"
            size="lg"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" /> 분석 중...
              </>
            ) : (
              '판별하기'
            )}
          </Button>
          {(file || result) && (
            <Button
              onClick={reset}
              variant="outline"
              size="lg"
              disabled={isLoading}
            >
              초기화
            </Button>
          )}
        </div>

        {result && (
          <div className="flex flex-col gap-4 rounded-lg border border-border/60 bg-background/40 p-5">
            <div className="flex items-center gap-3">
              {isReal ? (
                <ShieldCheck className="size-7 text-emerald-500" />
              ) : (
                <ShieldAlert className="size-7 text-rose-500" />
              )}
              <div className="flex flex-col">
                <div className="text-xl font-bold tracking-tight">
                  {isReal ? '실제 음성 (Real)' : '합성 음성 (Fake)'}
                </div>
                {result.inference_ms !== undefined && (
                  <div className="text-xs text-muted-foreground">
                    추론 시간 {result.inference_ms.toFixed(0)} ms
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <div>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-muted-foreground">실제 음성 확률</span>
                  <span className="font-mono font-medium">{realPct}%</span>
                </div>
                <Progress
                  value={realPct}
                  indicatorClassName="bg-emerald-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1.5">
                  <span className="text-muted-foreground">합성 음성 확률</span>
                  <span className="font-mono font-medium">{fakePct}%</span>
                </div>
                <Progress value={fakePct} indicatorClassName="bg-rose-500" />
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
