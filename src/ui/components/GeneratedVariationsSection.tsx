import { useState } from 'react';
import { Stack, Text, Card, Button, Spinner, Modal, Flex } from '@/kui-foundations-react-external';
import { ModelViewer3D } from '@/components/ModelViewer3D';

interface Props {
  generatedImages: (string | null)[];
  qualityScores: (number | null)[];
  qualityIssues: (string[] | null)[];
  generated3DModel: string | null;
  model3DError: string | null;
  isGenerating: boolean;
}

export function GeneratedVariationsSection({ 
  generatedImages,
  qualityScores,
  qualityIssues,
  generated3DModel, 
  model3DError,
  isGenerating 
}: Props) {
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(null);

  const downloadImage = (imageUrl: string, index: number) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `generated-variation-${index + 1}.png`;
    link.click();
  };

  const download3DModel = (modelUrl: string) => {
    const link = document.createElement('a');
    link.href = modelUrl;
    link.download = 'generated-model.glb';
    link.click();
  };

  const openImageModal = (index: number) => {
    if (generatedImages[index]) {
      setSelectedImageIndex(index);
      setModalOpen(true);
    }
  };

  const getScoreBadgeColor = (score: number): string => {
    if (score < 40) {
      return '#ef4444';
    } else if (score < 80) {
      return '#f97316';
    } else {
      return '#76B900';
    }
  };

  return (
    <div style={{ padding: '0 16px' }}>
      <Stack gap="6">
        <Text kind="title/lg" style={{ color: 'white' }}>Generated Image Variations</Text>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
          {[0, 1].map((index) => (
            <Card key={index}>
              <Stack gap="4">
                {generatedImages[index] ? (
                  <div 
                    className="rounded-lg overflow-hidden"
                    style={{ 
                      minHeight: '300px',
                      backgroundColor: 'var(--color-gray-1000)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      position: 'relative'
                    }}
                    onClick={() => openImageModal(index)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openImageModal(index);
                      }
                    }}
                  >
                    <img 
                      src={generatedImages[index]!} 
                      alt={`Generated variation ${index + 1}`}
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '300px',
                        width: 'auto',
                        height: 'auto',
                        objectFit: 'contain',
                        display: 'block',
                        transition: 'transform 0.2s ease-in-out'
                      }}
                      onMouseOver={(e) => {
                        (e.target as HTMLImageElement).style.transform = 'scale(1.05)';
                      }}
                      onMouseOut={(e) => {
                        (e.target as HTMLImageElement).style.transform = 'scale(1)';
                      }}
                    />
                    <div
                      style={{
                        position: 'absolute',
                        top: '12px',
                        right: '12px',
                        backgroundColor: 'var(--background-color-surface-overlay)',
                        borderRadius: 'var(--radius-md)',
                        padding: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        opacity: 0.9,
                        pointerEvents: 'none'
                      }}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-color-primary)">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
                      </svg>
                    </div>
                    {qualityScores[index] !== null && (
                      <div
                        style={{
                          position: 'absolute',
                          top: '12px',
                          left: '12px',
                          backgroundColor: getScoreBadgeColor(qualityScores[index]!),
                          borderRadius: 'var(--radius-md)',
                          padding: '6px 12px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          opacity: 0.95,
                          pointerEvents: 'none',
                          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
                        }}
                      >
                        <Text 
                          kind="body/bold/sm" 
                          style={{ 
                            color: 'white',
                            fontSize: '13px',
                            lineHeight: '1'
                          }}
                        >
                          {qualityScores[index]!.toFixed(1)}%
                        </Text>
                      </div>
                    )}
                  </div>
                ) : (
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
                        <Spinner size="large" description={`Generating variation ${index + 1}...`} />
                      </Stack>
                    ) : (
                      <Stack gap="3" align="center">
                        <div className="w-16 h-16 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <Text kind="body/regular/sm" className="text-subtle">Variation {index + 1}</Text>
                      </Stack>
                    )}
                  </div>
                )}
                {generatedImages[index] && qualityIssues[index] && qualityIssues[index]!.length > 0 && (
                  <div
                    style={{
                      padding: '12px',
                      backgroundColor: 'var(--background-color-surface-sunken)',
                      borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-color-base)'
                    }}
                  >
                    <Stack gap="2">
                      <Text 
                        kind="body/bold/xs" 
                        style={{ 
                          color: 'var(--text-color-subtle)',
                          textTransform: 'uppercase',
                          letterSpacing: '0.5px',
                          fontSize: '11px'
                        }}
                      >
                        Quality Issues
                      </Text>
                      <Stack gap="1">
                        {qualityIssues[index]!.map((issue, issueIdx) => (
                          <div
                            key={issueIdx}
                            style={{
                              display: 'flex',
                              alignItems: 'flex-start',
                              gap: '6px'
                            }}
                          >
                            <span style={{ 
                              color: '#f97316',
                              fontSize: '14px',
                              lineHeight: '1.4',
                              marginTop: '1px'
                            }}>•</span>
                            <Text 
                              kind="body/regular/xs" 
                              style={{ 
                                color: 'var(--text-color-primary)',
                                fontSize: '12px',
                                lineHeight: '1.4',
                                flex: 1
                              }}
                            >
                              {issue}
                            </Text>
                          </div>
                        ))}
                      </Stack>
                    </Stack>
                  </div>
                )}
                <Button 
                  kind="secondary" 
                  size="medium" 
                  disabled={!generatedImages[index]}
                  onClick={() => generatedImages[index] && downloadImage(generatedImages[index]!, index)}
                >
                  Download
                </Button>
              </Stack>
            </Card>
          ))}

          <Card>
            <Stack gap="4">
              <ModelViewer3D 
                modelUrl={generated3DModel}
                error={model3DError}
                isGenerating={isGenerating}
              />
              <Button 
                kind="secondary" 
                size="medium" 
                disabled={!generated3DModel}
                onClick={() => generated3DModel && download3DModel(generated3DModel)}
              >
                Download GLB
              </Button>
            </Stack>
          </Card>
        </div>

        <Modal
          open={modalOpen}
          onOpenChange={setModalOpen}
          slotHeading={
            selectedImageIndex !== null 
              ? `Generated Variation ${selectedImageIndex + 1}` 
              : 'Generated Image'
          }
          slotFooter={
            <Flex align="center" justify="end" gap="3">
              <Button 
                kind="tertiary" 
                size="medium"
                onClick={() => setModalOpen(false)}
              >
                Close
              </Button>
              {selectedImageIndex !== null && generatedImages[selectedImageIndex] && (
                <Button 
                  kind="primary" 
                  size="medium"
                  onClick={() => {
                    if (selectedImageIndex !== null && generatedImages[selectedImageIndex]) {
                      downloadImage(generatedImages[selectedImageIndex]!, selectedImageIndex);
                    }
                  }}
                >
                  Download
                </Button>
              )}
            </Flex>
          }
          density="spacious"
          style={{ maxWidth: '900px', width: 'auto' }}
        >
          <div style={{ 
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'var(--background-color-surface-sunken)',
            borderRadius: 'var(--radius-md)',
            padding: '24px'
          }}>
            {selectedImageIndex !== null && generatedImages[selectedImageIndex] && (
              <img 
                src={generatedImages[selectedImageIndex]!} 
                alt={`Generated variation ${selectedImageIndex + 1}`}
                style={{ 
                  width: 'auto',
                  height: 'auto',
                  maxWidth: '100%',
                  maxHeight: '60vh',
                  objectFit: 'contain',
                  display: 'block'
                }}
              />
            )}
          </div>
        </Modal>
      </Stack>
    </div>
  );
}

