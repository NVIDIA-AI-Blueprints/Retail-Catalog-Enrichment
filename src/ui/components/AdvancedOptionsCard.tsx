import { Card, Stack, Text, Button, Flex, Switch, TextArea } from '@/kui-foundations-react-external';
import { PolicyDocument, PolicyUploadResult } from '@/types';
import { useState, useEffect, CSSProperties } from 'react';

const UPLOAD_STAGES = [
  'Uploading PDFs',
  'Parsing documents',
  'Building embeddings',
  'Indexing vectors',
] as const;

const innerCardStyle: CSSProperties = {
  background: 'var(--color-surface-sunken)',
  border: '1px solid var(--color-border-base)',
  borderRadius: '18px',
  padding: '18px',
};

const toggleRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '16px',
  padding: '14px 16px',
  border: '1px solid var(--color-border-base)',
  borderRadius: '14px',
  background: 'var(--color-surface-base)',
};

const pillStyle: CSSProperties = {
  border: '1px solid var(--color-border-base)',
  background: 'rgba(255,255,255,0.03)',
  padding: '6px 10px',
  borderRadius: '999px',
  fontSize: '12px',
  color: 'var(--text-color-subtle)',
  whiteSpace: 'nowrap',
  flexShrink: 0,
};

interface Props {
  brandInstructions: string;
  loadedPolicies: PolicyDocument[];
  policyUploadResults: PolicyUploadResult[];
  policyUploadError: string | null;
  isUploadingPolicies: boolean;
  enableVariation1: boolean;
  enableVariation2: boolean;
  enable3D: boolean;
  enableWebInsights: boolean;
  isAnalyzingFields: boolean;
  isGeneratingImage: boolean;
  manualFilename: string | null;
  manualChunkCount: number | null;
  isUploadingManual: boolean;
  manualUploadError: string | null;
  onBrandInstructionsChange: (value: string) => void;
  onPolicyFileSelect: () => void;
  onClearPolicyLibrary: () => void;
  onManualFileSelect: () => void;
  onClearManual: () => void;
  onEnableVariation1Change: (value: boolean) => void;
  onEnableVariation2Change: (value: boolean) => void;
  onEnable3DChange: (value: boolean) => void;
  onEnableWebInsightsChange: (value: boolean) => void;
}

const MANUAL_UPLOAD_STAGES = [
  'Uploading PDF',
  'Extracting text',
  'Building embeddings',
  'Generating queries',
  'Retrieving knowledge',
] as const;

function StagedUploadProgress({ stages }: { stages: readonly string[] }) {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex((prev) => (prev + 1) % stages.length);
    }, 2400);
    return () => clearInterval(interval);
  }, [stages.length]);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
        padding: '16px 20px',
        border: '1px solid rgba(118, 185, 0, 0.25)',
        borderRadius: '14px',
        background: 'rgba(118, 185, 0, 0.06)',
        marginTop: '14px',
      }}
    >
      <div
        style={{
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          border: '2.5px solid rgba(118, 185, 0, 0.15)',
          borderTopColor: '#76B900',
          animation: 'upload-spin 0.8s linear infinite',
          flexShrink: 0,
        }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <Text kind="body/semibold/sm" style={{ color: '#76B900' }}>
            {stages[stageIndex]}
          </Text>
          <span style={{ color: 'rgba(118, 185, 0, 0.6)', fontSize: '12px', animation: 'upload-dots 1.4s steps(4, end) infinite' }}>
            ...
          </span>
        </div>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {stages.map((label, i) => (
            <div
              key={label}
              style={{
                width: i <= stageIndex ? '20px' : '6px',
                height: '3px',
                borderRadius: '2px',
                background: i <= stageIndex ? '#76B900' : 'rgba(255,255,255,0.1)',
                transition: 'all 0.4s ease',
              }}
            />
          ))}
        </div>
      </div>
      <style>{`
        @keyframes upload-spin {
          to { transform: rotate(360deg); }
        }
        @keyframes upload-dots {
          0% { opacity: 0.2; }
          50% { opacity: 1; }
          100% { opacity: 0.2; }
        }
      `}</style>
    </div>
  );
}

export function AdvancedOptionsCard({
  brandInstructions,
  loadedPolicies,
  policyUploadResults,
  policyUploadError,
  isUploadingPolicies,
  enableVariation1,
  enableVariation2,
  enable3D,
  enableWebInsights,
  isAnalyzingFields,
  isGeneratingImage,
  manualFilename,
  manualChunkCount,
  isUploadingManual,
  manualUploadError,
  onBrandInstructionsChange,
  onPolicyFileSelect,
  onClearPolicyLibrary,
  onManualFileSelect,
  onClearManual,
  onEnableVariation1Change,
  onEnableVariation2Change,
  onEnable3DChange,
  onEnableWebInsightsChange,
}: Props) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const disabled = isAnalyzingFields || isGeneratingImage;

  return (
    <Card>
      <Flex justify="between" align="start">
        <Stack gap="2">
          <Text kind="body/regular/sm" className="text-subtle" style={{ letterSpacing: '0.08em', textTransform: 'uppercase', fontSize: '12px' }}>
            Configuration
          </Text>
          <Text kind="title/md" className="text-primary">Advanced options</Text>
          <Text kind="body/regular/sm" className="text-subtle">
            Fine-tune generation with brand rules, compliance policies, and output toggles.
          </Text>
        </Stack>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? 'Expand section' : 'Collapse section'}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '999px',
            display: 'grid',
            placeItems: 'center',
            border: '1px solid var(--color-border-base)',
            background: 'var(--color-surface-raised)',
            color: 'var(--text-color-subtle)',
            cursor: 'pointer',
            transition: '0.2s ease',
            flexShrink: 0,
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            style={{
              transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
            }}
          >
            <path d="M4 10L8 6L12 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </Flex>

      {!isCollapsed && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 0.95fr', gap: '24px', marginTop: '24px' }}>
          <Stack gap="5">
            {/* Brand Instructions */}
            <div style={innerCardStyle}>
              <Flex justify="between" align="start" style={{ marginBottom: '12px' }}>
                <Stack gap="1">
                  <Text kind="body/semibold/md" className="text-primary">Brand instructions</Text>
                  <Text kind="body/regular/sm" className="text-subtle">
                    Add voice, tone, taxonomy, or content rules that should guide generation.
                  </Text>
                </Stack>
                <span style={pillStyle}>Optional</span>
              </Flex>
              <TextArea
                placeholder="Example: Use a clean enterprise tone, avoid marketing fluff, keep category labels consistent, and prefer concise product descriptions."
                size="medium"
                resizeable="manual"
                value={brandInstructions}
                onChange={(e: any) => onBrandInstructionsChange(e.target.value)}
                disabled={disabled}
                attributes={{ TextAreaElement: { rows: 5 } }}
              />
            </div>

            {/* Policy Library */}
            <div style={innerCardStyle}>
              <Flex justify="between" align="center" style={{ marginBottom: '14px' }}>
                <Stack gap="1">
                  <Text kind="body/semibold/md" className="text-primary">Policy library</Text>
                  <Text kind="body/regular/sm" className="text-subtle">
                    Upload reference PDFs or start with an empty workspace.
                  </Text>
                </Stack>
                {loadedPolicies.length > 0 ? (
                  <Text kind="body/regular/sm" className="text-subtle" style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>
                    {loadedPolicies.length} {loadedPolicies.length === 1 ? 'file' : 'files'} loaded
                  </Text>
                ) : (
                  <span style={pillStyle}>Optional</span>
                )}
              </Flex>

              {!isUploadingPolicies && (
                <Flex gap="3" align="center">
                  <Button
                    kind="primary"
                    size="medium"
                    className="nvidia-green-button"
                    onClick={onPolicyFileSelect}
                    disabled={disabled}
                  >
                    Upload PDFs
                  </Button>
                  {loadedPolicies.length > 0 && (
                    <Button
                      kind="secondary"
                      size="medium"
                      onClick={onClearPolicyLibrary}
                      disabled={disabled}
                    >
                      Start from scratch
                    </Button>
                  )}
                </Flex>
              )}

              {isUploadingPolicies && <StagedUploadProgress stages={UPLOAD_STAGES} />}

              {policyUploadError && (
                <div className="p-3 rounded-lg border" style={{ borderColor: 'var(--color-red-500)', backgroundColor: 'rgba(255, 84, 89, 0.08)', marginTop: '14px' }}>
                  <Text kind="body/regular/sm" style={{ color: 'var(--color-red-400)' }}>
                    {policyUploadError}
                  </Text>
                </div>
              )}

              {policyUploadResults.length > 0 && !isUploadingPolicies && (
                <div className="p-3 rounded-lg border border-base bg-surface-sunken" style={{ marginTop: '14px' }}>
                  <Text kind="body/regular/sm" className="text-primary">
                    {policyUploadResults.filter((r) => !r.already_loaded).length} new,{" "}
                    {policyUploadResults.filter((r) => r.already_loaded).length} already loaded
                  </Text>
                </div>
              )}

              {loadedPolicies.length > 0 && !isUploadingPolicies && (
                <Stack gap="3" style={{ marginTop: '14px' }}>
                  {loadedPolicies.map((doc) => (
                    <div
                      key={doc.document_hash}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '14px',
                        padding: '14px',
                        border: '1px solid var(--color-border-base)',
                        borderRadius: '14px',
                        background: 'var(--color-surface-base)',
                      }}
                    >
                      <Flex gap="3" align="center" style={{ minWidth: 0 }}>
                        <div
                          style={{
                            width: '42px',
                            height: '42px',
                            borderRadius: '12px',
                            background: 'rgba(255,255,255,0.05)',
                            display: 'grid',
                            placeItems: 'center',
                            fontSize: '18px',
                            flexShrink: 0,
                          }}
                        >
                          📄
                        </div>
                        <Stack gap="1" style={{ minWidth: 0 }}>
                          <Text kind="body/semibold/sm" className="text-primary" style={{ wordBreak: 'break-word' }}>
                            {doc.filename}
                          </Text>
                          <Text kind="body/regular/sm" className="text-subtle">
                            {doc.chunk_count} indexed {doc.chunk_count === 1 ? 'record' : 'records'}
                          </Text>
                        </Stack>
                      </Flex>
                      <span style={pillStyle}>Ready</span>
                    </div>
                  ))}
                </Stack>
              )}
            </div>

            {/* Product Manual for FAQs */}
            <div style={innerCardStyle}>
              <Flex justify="between" align="center" style={{ marginBottom: '14px' }}>
                <Stack gap="1">
                  <Text kind="body/semibold/md" className="text-primary">Product manual for FAQs</Text>
                  <Text kind="body/regular/sm" className="text-subtle">
                    Upload a product manual PDF to generate richer FAQs with specs, care instructions, and more.
                  </Text>
                </Stack>
                <span style={pillStyle}>Optional</span>
              </Flex>

              {!isUploadingManual && !manualFilename && (
                <Button
                  kind="primary"
                  size="medium"
                  className="nvidia-green-button"
                  onClick={onManualFileSelect}
                  disabled={disabled}
                >
                  Upload PDF
                </Button>
              )}

              {isUploadingManual && <StagedUploadProgress stages={MANUAL_UPLOAD_STAGES} />}

              {manualUploadError && (
                <div className="p-3 rounded-lg border" style={{ borderColor: 'var(--color-red-500)', backgroundColor: 'rgba(255, 84, 89, 0.08)', marginTop: '14px' }}>
                  <Text kind="body/regular/sm" style={{ color: 'var(--color-red-400)' }}>
                    {manualUploadError}
                  </Text>
                </div>
              )}

              {manualFilename && !isUploadingManual && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '14px',
                    padding: '14px',
                    border: '1px solid var(--color-border-base)',
                    borderRadius: '14px',
                    background: 'var(--color-surface-base)',
                    marginTop: '14px',
                  }}
                >
                  <Flex gap="3" align="center" style={{ minWidth: 0 }}>
                    <div
                      style={{
                        width: '42px',
                        height: '42px',
                        borderRadius: '12px',
                        background: 'rgba(255,255,255,0.05)',
                        display: 'grid',
                        placeItems: 'center',
                        fontSize: '18px',
                        flexShrink: 0,
                      }}
                    >
                      📋
                    </div>
                    <Stack gap="1" style={{ minWidth: 0 }}>
                      <Text kind="body/semibold/sm" className="text-primary" style={{ wordBreak: 'break-word' }}>
                        {manualFilename}
                      </Text>
                      <Text kind="body/regular/sm" className="text-subtle">
                        {manualChunkCount} indexed {manualChunkCount === 1 ? 'chunk' : 'chunks'}
                      </Text>
                    </Stack>
                  </Flex>
                  <Button
                    kind="secondary"
                    size="small"
                    onClick={onClearManual}
                    disabled={disabled}
                  >
                    Remove
                  </Button>
                </div>
              )}
            </div>
          </Stack>

          {/* Image Options */}
          <Stack gap="5">
            <div style={innerCardStyle}>
              <Stack gap="1" style={{ marginBottom: '14px' }}>
                <Text kind="body/semibold/md" className="text-primary">Image options</Text>
                <Text kind="body/regular/sm" className="text-subtle">
                  Toggle which generation outputs to include.
                </Text>
              </Stack>

              <Stack gap="3">
                <div style={toggleRowStyle}>
                  <Stack gap="1">
                    <Text kind="body/semibold/sm" className="text-primary">Image variation 1</Text>
                    <Text kind="body/regular/sm" className="text-subtle" style={{ fontSize: '12px' }}>
                      Generate an alternate visual based on the source image.
                    </Text>
                  </Stack>
                  <Switch checked={enableVariation1} onCheckedChange={onEnableVariation1Change} disabled={disabled} />
                </div>

                <div style={toggleRowStyle}>
                  <Stack gap="1">
                    <Text kind="body/semibold/sm" className="text-primary">Image variation 2</Text>
                    <Text kind="body/regular/sm" className="text-subtle" style={{ fontSize: '12px' }}>
                      Enable a second variation path for broader creative output.
                    </Text>
                  </Stack>
                  <Switch checked={enableVariation2} onCheckedChange={onEnableVariation2Change} disabled={disabled} />
                </div>

                <div style={toggleRowStyle}>
                  <Stack gap="1">
                    <Text kind="body/semibold/sm" className="text-primary">3D model</Text>
                    <Text kind="body/regular/sm" className="text-subtle" style={{ fontSize: '12px' }}>
                      Allow generation workflows that output 3D assets.
                    </Text>
                  </Stack>
                  <Switch checked={enable3D} onCheckedChange={onEnable3DChange} disabled={disabled} />
                </div>
              </Stack>
            </div>

            <div style={innerCardStyle}>
              <Stack gap="1" style={{ marginBottom: '14px' }}>
                <Text kind="body/semibold/md" className="text-primary">Research options</Text>
                <Text kind="body/regular/sm" className="text-subtle">
                  Toggle optional source-backed enrichment.
                </Text>
              </Stack>

              <Stack gap="3">
                <div style={toggleRowStyle}>
                  <Stack gap="1">
                    <Text kind="body/semibold/sm" className="text-primary">Web Insights</Text>
                    <Text kind="body/regular/sm" className="text-subtle" style={{ fontSize: '12px' }}>
                      Research source-backed product context with Exa.
                    </Text>
                  </Stack>
                  <Switch checked={enableWebInsights} onCheckedChange={onEnableWebInsightsChange} disabled={disabled} />
                </div>
              </Stack>
            </div>
          </Stack>
        </div>
      )}
    </Card>
  );
}
