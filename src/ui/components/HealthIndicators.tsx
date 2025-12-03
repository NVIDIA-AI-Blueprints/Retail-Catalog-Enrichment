'use client';

import { Tooltip, Flex } from '@/kui-foundations-react-external';
import { NIMHealthStatus, HealthState } from '../types';

interface HealthIndicatorsProps {
  health: NIMHealthStatus;
}

interface ServiceIndicatorProps {
  name: string;
  status: HealthState;
}

function ServiceIndicator({ name, status }: ServiceIndicatorProps) {
  const getColor = () => {
    if (status === 'healthy') return '#10b981'; // green
    if (status === 'unhealthy') return '#ef4444'; // red
    return '#fbbf24'; // yellow for checking
  };

  const message = status === 'healthy' 
    ? `${name} is healthy`
    : status === 'unhealthy'
    ? `${name} is unhealthy`
    : `${name} is checking...`;

  return (
    <Tooltip slotContent={message}>
      <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
        <div
          style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: getColor(),
            boxShadow: `0 0 4px ${getColor()}`,
          }}
        />
      </div>
    </Tooltip>
  );
}

export function HealthIndicators({ health }: HealthIndicatorsProps) {
  return (
    <Flex gap="2" align="center">
      <ServiceIndicator name="VLM" status={health.vlm} />
      <ServiceIndicator name="LLM" status={health.llm} />
      <ServiceIndicator name="FLUX" status={health.flux} />
      <ServiceIndicator name="TRELLIS" status={health.trellis} />
    </Flex>
  );
}

