import { Card, Stack, Text, Button, Flex, Spinner } from '@/kui-foundations-react-external';
import { ImageMetadata } from '@/types';

interface Props {
  uploadedImage: string | null;
  isUploading: boolean;
  imageMetadata: ImageMetadata | null;
  onFileSelect: () => void;
  onFileChange: () => void;
  onFileRemove: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
}

export function ImageUploadCard({ 
  uploadedImage, 
  isUploading, 
  imageMetadata, 
  onFileSelect,
  onFileChange,
  onFileRemove,
  onDragOver,
  onDrop 
}: Props) {
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
              <div 
                style={{
                  position: 'absolute',
                  bottom: '12px',
                  left: '0',
                  right: '0',
                  display: 'flex',
                  gap: '8px',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}
              >
                <Button kind="secondary" size="small" onClick={onFileChange}>
                  Change
                </Button>
                <Button kind="secondary" size="small" onClick={onFileRemove}>
                  Remove
                </Button>
              </div>
            </div>
            
            {imageMetadata && (
              <div className="bg-surface-sunken rounded-lg p-4 border border-base">
                <Stack gap="2">
                  <Flex justify="between" align="center">
                    <Text kind="body/regular/sm" className="text-subtle">File Name</Text>
                    <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.name}</Text>
                  </Flex>
                  <Flex justify="between" align="center">
                    <Text kind="body/regular/sm" className="text-subtle">Size</Text>
                    <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.size}</Text>
                  </Flex>
                  {imageMetadata.dimensions && (
                    <Flex justify="between" align="center">
                      <Text kind="body/regular/sm" className="text-subtle">Dimensions</Text>
                      <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.dimensions}</Text>
                    </Flex>
                  )}
                </Stack>
              </div>
            )}
          </>
        ) : (
          <div 
            className="border-2 border-dashed border-base rounded-lg p-16 text-center bg-surface-sunken hover:bg-surface-raised transition-colors cursor-pointer"
            onClick={onFileSelect}
            onDragOver={onDragOver}
            onDrop={onDrop}
          >
            <Stack gap="4" align="center">
              <div className="w-20 h-20 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <Stack gap="2" align="center">
                <Text kind="body/semibold/md" className="text-primary">
                  Click to upload or drag and drop
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

