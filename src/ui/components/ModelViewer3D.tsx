import { Stack, Text, Spinner } from '@/kui-foundations-react-external';
import { useEffect, useRef, useState } from 'react';

interface Props {
  modelUrl: string | null;
  error: string | null;
  isGenerating: boolean;
}

export function ModelViewer3D({ modelUrl, error, isGenerating }: Props) {
  const modelViewerRef = useRef<HTMLElement | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const viewer = modelViewerRef.current;
    if (!viewer || !modelUrl) return;

    setIsLoading(true);
    
    const handleLoad = () => {
      setIsLoading(false);
    };

    const handleError = () => {
      setIsLoading(false);
    };

    viewer.addEventListener('load', handleLoad);
    viewer.addEventListener('error', handleError);
    
    return () => {
      viewer.removeEventListener('load', handleLoad);
      viewer.removeEventListener('error', handleError);
    };
  }, [modelUrl]);

  if (error) {
    return (
      <div 
        className="bg-surface-sunken rounded-lg border-2 border-dashed border-base"
        style={{ 
          minHeight: '300px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '16px'
        }}
      >
        <Stack gap="3" align="center">
          <div className="w-16 h-16 bg-red-900/20 rounded-lg flex items-center justify-center border border-red-500">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--color-red-500)">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <Text kind="body/regular/sm" className="text-center" style={{ color: 'var(--color-red-500)' }}>
            {error}
          </Text>
        </Stack>
      </div>
    );
  }

  if (modelUrl) {
    return (
      <div 
        className="rounded-lg overflow-hidden model-viewer-container"
        style={{ 
          minHeight: '300px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}
      >
        {/* @ts-ignore - model-viewer is a custom element */}
        <model-viewer
          ref={modelViewerRef}
          src={modelUrl}
          alt="Generated 3D model"
          camera-controls="true"
          auto-rotate="true"
          shadow-intensity="1"
          exposure="1"
          style={{
            width: '100%',
            height: '300px',
            backgroundColor: 'transparent',
            position: 'relative',
            zIndex: 2
          }}
        />
        {isLoading && (
          <div 
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(12, 12, 12, 0.8)',
              zIndex: 10
            }}
          >
            <Stack gap="3" align="center">
              <Spinner size="large" description="Loading 3D model..." />
            </Stack>
          </div>
        )}
      </div>
    );
  }

  return (
    <div 
      className="bg-surface-sunken rounded-lg border-2 border-dashed border-base"
      style={{ 
        minHeight: '300px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {isGenerating ? (
        <Stack gap="3" align="center">
          <Spinner size="large" description="Generating 3D model..." />
        </Stack>
      ) : (
        <Stack gap="3" align="center">
          <div className="w-16 h-16 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          </div>
          <Text kind="body/regular/sm" className="text-subtle">3D Model</Text>
        </Stack>
      )}
    </div>
  );
}

