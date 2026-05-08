'use client';

import { useEffect, useRef, useState } from 'react';

type AnimatedCounterProps = {
  value: number;
  durationMs?: number;
  decimals?: number;
};

export function AnimatedCounter({
  value,
  durationMs = 800,
  decimals = 0,
}: AnimatedCounterProps) {
  const [display, setDisplay] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const startValueRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    startTimeRef.current = null;
    startValueRef.current = display;

    const tick = (now: number) => {
      if (startTimeRef.current === null) startTimeRef.current = now;
      const elapsed = now - startTimeRef.current;
      const t = Math.min(elapsed / durationMs, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const next =
        startValueRef.current + (value - startValueRef.current) * eased;
      setDisplay(next);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, durationMs]);

  return <>{display.toFixed(decimals)}</>;
}
