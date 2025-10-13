import { Card, Stack, Text, Button, Flex, Spinner, Select } from '@/kui-foundations-react-external';
import { ImageMetadata, LocaleOption } from '@/types';
import { useState } from 'react';

interface Props {
  uploadedImage: string | null;
  isUploading: boolean;
  imageMetadata: ImageMetadata | null;
  locale: string;
  localeOptions: LocaleOption[];
  isAnalyzingFields: boolean;
  isGeneratingImage: boolean;
  onFileSelect: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onLocaleChange: (value: string) => void;
  onGenerate: () => void;
  onReset: () => void;
}

export function ImageUploadCard({ 
  uploadedImage, 
  isUploading, 
  imageMetadata, 
  locale,
  localeOptions,
  isAnalyzingFields,
  isGeneratingImage,
  onFileSelect,
  onDragOver,
  onDrop,
  onLocaleChange,
  onGenerate,
  onReset
}: Props) {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    onDragOver(e);
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    onDrop(e);
    setIsDragging(false);
  };

  return (
    <Card>
      <Stack gap="6">
        <Flex justify="between" align="center">
          <Text kind="title/md" className="text-primary">Image</Text>
          {uploadedImage && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-300" style={{ backgroundColor: 'var(--color-green-300)' }}></div>
              <Text kind="body/regular/sm" className="text-subtle">Ready</Text>
            </div>
          )}
        </Flex>
        
        {isUploading ? (
          <div className="border-2 border-dashed border-base rounded-lg p-16 text-center bg-surface-sunken">
            <Stack gap="4" align="center">
              <Spinner size="large" aria-label="Uploading image..." />
            </Stack>
          </div>
        ) : uploadedImage ? (
          <>
            <div 
              className="relative rounded-lg overflow-hidden nvidia-green-border" 
              style={{ 
                minHeight: '400px',
                backgroundColor: 'var(--color-gray-1000)',
                borderWidth: '2px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <img 
                src={uploadedImage} 
                alt="Uploaded preview" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '400px',
                  width: 'auto',
                  height: 'auto',
                  objectFit: 'contain',
                  display: 'block'
                }}
              />
            </div>
            
            {/* Controls moved below the image */}
            <Flex gap="3" align="center">
              <Button 
                kind="primary" 
                size="large" 
                className="nvidia-green-button"
                disabled={!uploadedImage || isAnalyzingFields || isGeneratingImage}
                onClick={onGenerate}
                style={{ flex: '1' }}
              >
                {isAnalyzingFields ? (
                  <Flex gap="2" align="center">
                    <Spinner size="small" aria-label="Analyzing" />
                    <span>Analyzing...</span>
                  </Flex>
                ) : isGeneratingImage ? (
                  <Flex gap="2" align="center">
                    <Spinner size="small" aria-label="Generating" />
                    <span>Generating Image...</span>
                  </Flex>
                ) : 'Generate Enriched Data'}
              </Button>
              
              <div style={{ minWidth: '160px' }}>
                <Select
                  items={localeOptions}
                  value={locale}
                  onValueChange={onLocaleChange}
                  placeholder="Select locale"
                  size="large"
                  disabled={isAnalyzingFields || isGeneratingImage}
                />
              </div>

              <Button 
                kind="secondary" 
                size="large"
                disabled={isAnalyzingFields || isGeneratingImage}
                onClick={onReset}
              >
                Reset
              </Button>
            </Flex>
          </>
        ) : (
          <div 
            className="rounded-lg text-center transition-all duration-200"
            style={{
              border: isDragging 
                ? '3px dashed #76B900' 
                : isHovered 
                  ? '2px dashed #76B900' 
                  : '2px dashed var(--color-border-base)',
              backgroundColor: isDragging 
                ? 'rgba(118, 185, 0, 0.08)' 
                : isHovered 
                  ? 'rgba(118, 185, 0, 0.05)' 
                  : 'var(--color-surface-sunken)',
              padding: '64px',
              cursor: 'pointer',
              transform: isDragging ? 'scale(1.01)' : 'scale(1)',
            }}
            onClick={onFileSelect}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onFileSelect();
              }
            }}
            aria-label="Upload image by clicking or dragging and dropping"
          >
            <Stack gap="4" align="center">
              <div 
                className="flex items-center justify-center rounded-lg border transition-all duration-200"
                style={{
                  width: '80px',
                  height: '80px',
                  backgroundColor: isDragging 
                    ? 'rgba(118, 185, 0, 0.12)' 
                    : isHovered 
                      ? 'rgba(118, 185, 0, 0.08)' 
                      : 'var(--color-surface-raised)',
                  borderColor: isDragging 
                    ? '#76B900' 
                    : isHovered 
                      ? 'rgba(118, 185, 0, 0.5)' 
                      : 'var(--color-border-base)',
                  transform: isHovered ? 'translateY(-4px)' : 'translateY(0)',
                }}
              >
                <svg 
                  className="transition-all duration-200" 
                  style={{
                    width: '40px',
                    height: '40px',
                    transform: isDragging ? 'scale(1.1)' : 'scale(1)',
                  }}
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke={isDragging || isHovered ? '#76B900' : 'var(--text-color-subtle)'}
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={1.5} 
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
                  />
                </svg>
              </div>
              <Stack gap="2" align="center">
                <Text 
                  kind="body/semibold/md" 
                  style={{
                    color: isDragging || isHovered 
                      ? '#76B900' 
                      : 'var(--text-color-primary)',
                    transition: 'color 0.2s'
                  }}
                >
                  {isDragging ? 'Drop your image here' : 'Click to upload or drag and drop'}
                </Text>
                <Text kind="body/regular/sm" className="text-subtle">
                  PNG, JPG or JPEG (max. 10MB)
                </Text>
              </Stack>
            </Stack>
          </div>
        )}
      </Stack>
    </Card>
  );
}

