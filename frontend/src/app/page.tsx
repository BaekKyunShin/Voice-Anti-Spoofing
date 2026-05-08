import { DottedSurface } from '@/components/ui/dotted-surface';
import { UploadForm } from '@/components/upload-form';
import { cn } from '@/lib/utils';

export default function Home() {
  return (
    <main className="relative flex flex-1 items-center justify-center px-4 py-16 overflow-hidden">
      <DottedSurface className="size-full" />
      <div
        aria-hidden="true"
        className={cn(
          'pointer-events-none absolute -top-10 left-1/2 size-full -translate-x-1/2 rounded-full',
          'bg-[radial-gradient(ellipse_at_center,--theme(--color-foreground/.1),transparent_50%)]',
          'blur-[30px]'
        )}
      />
      <div className="relative z-10 w-full flex justify-center">
        <UploadForm />
      </div>
    </main>
  );
}
