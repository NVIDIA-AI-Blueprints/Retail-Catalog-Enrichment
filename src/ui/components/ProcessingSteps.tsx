import { Stack, Text, Spinner } from '@/kui-foundations-react-external';
import { useEffect, useState } from 'react';

type StepStatus = 'pending' | 'active' | 'complete' | 'error';

interface Step {
  id: string;
  label: string;
  description: string;
  status: StepStatus;
}

interface Props {
  isAnalyzing: boolean;
  hasAugmentedData: boolean;
}

const STEP_DEFINITIONS = [
  { id: 'vlm', label: 'Vision Analysis', description: 'Extracting visual features and attributes' },
  { id: 'llm', label: 'Content Enrichment', description: 'Enhancing product content with LLM, current data and brand instructions' },
];

const createSteps = (statuses: StepStatus[]): Step[] =>
  STEP_DEFINITIONS.map((def, i) => ({ ...def, status: statuses[i] }));

const StatusIcon = ({ status }: { status: StepStatus }) =>
  status === 'active' ? (
    <Spinner size="large" aria-label="Processing" />
  ) : status === 'complete' ? (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" fill="#76B900" fillOpacity="0.3"/>
      <path d="M7 12L10.5 15.5L17 9" stroke="#76B900" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ) : (
    <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '3px solid rgba(128, 128, 128, 0.4)' }} />
  );

const StepCard = ({ step }: { step: Step }) => (
  <div style={{
    flex: '1',
    padding: '24px',
    borderRadius: '12px',
    backgroundColor: step.status === 'active' ? 'rgba(118, 185, 0, 0.1)' : 
                     step.status === 'complete' ? 'rgba(118, 185, 0, 0.05)' : 
                     'rgba(255, 255, 255, 0.02)',
    border: `1px solid ${step.status === 'active' ? 'rgba(118, 185, 0, 0.4)' : 
            step.status === 'complete' ? 'rgba(118, 185, 0, 0.2)' : 'rgba(128, 128, 128, 0.2)'}`,
    transform: step.status === 'active' ? 'scale(1.02)' : 'scale(1)',
    transition: 'all 0.5s'
  }}>
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      <div style={{ flexShrink: 0 }}>
        <StatusIcon status={step.status} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <Text kind="body/semibold/lg" className={step.status === 'active' ? 'nvidia-green-text' : 'text-primary'}>
          {step.label}
        </Text>
        <div style={{ marginTop: '4px' }}>
          <Text kind="body/regular/sm" className="text-secondary" style={{ opacity: step.status === 'pending' ? 0.5 : 1 }}>
            {step.description}
          </Text>
        </div>
      </div>
    </div>
  </div>
);

const ArrowConnector = ({ isComplete }: { isComplete: boolean }) => (
  <div style={{ flexShrink: 0 }}>
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
      <path 
        d="M5 12H19M19 12L12 5M19 12L12 19" 
        stroke={isComplete ? '#76B900' : 'rgba(128, 128, 128, 0.4)'} 
        strokeWidth="2.5" 
        strokeLinecap="round" 
        strokeLinejoin="round"
        style={{ transition: 'all 0.5s' }}
      />
    </svg>
  </div>
);

export function ProcessingSteps({ isAnalyzing, hasAugmentedData }: Props) {
  const [steps, setSteps] = useState<Step[]>(createSteps(['pending', 'pending']));

  useEffect(() => {
    if (isAnalyzing && !hasAugmentedData) {
      setSteps(createSteps(['active', 'pending']));
      const timer = setTimeout(() => setSteps(createSteps(['complete', 'active'])), 2000);
      return () => clearTimeout(timer);
    }
    setSteps(createSteps(hasAugmentedData ? ['complete', 'complete'] : ['pending', 'pending']));
  }, [isAnalyzing, hasAugmentedData]);

  if (!isAnalyzing && !hasAugmentedData) return null;

  return (
    <div className="w-full py-6">
      <Stack gap="6" align="center">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px', width: '100%', maxWidth: '1000px', padding: '0 24px' }}>
          {steps.map((step, index) => (
            <div key={step.id} style={{ display: 'contents' }}>
              <StepCard step={step} />
              {index < steps.length - 1 && <ArrowConnector isComplete={step.status === 'complete'} />}
            </div>
          ))}
        </div>
      </Stack>
    </div>
  );
}