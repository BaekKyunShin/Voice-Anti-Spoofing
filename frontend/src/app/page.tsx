import { UploadForm } from '@/components/upload-form';

export default function Home() {
  return (
    <main
      className="relative flex flex-1 min-h-screen items-center justify-center px-4 py-8 overflow-hidden"
      style={{
        background:
          'radial-gradient(ellipse 110% 80% at 50% 30%,#0b1135 0%,#050508 100%)',
      }}
    >
      <UploadForm />
    </main>
  );
}
