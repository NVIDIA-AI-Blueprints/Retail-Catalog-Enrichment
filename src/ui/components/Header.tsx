'use client';

import { useEffect, useState } from 'react';
import { AppBar, Text, Flex } from '@/kui-foundations-react-external';
import Image from 'next/image';
import { HealthIndicators } from './HealthIndicators';
import { checkNIMHealth } from '../lib/api';
import { NIMHealthStatus } from '../types';

export function Header() {
  const [health, setHealth] = useState<NIMHealthStatus>({
    vlm: 'checking',
    llm: 'checking',
    flux: 'checking',
    trellis: 'checking'
  });

  useEffect(() => {
    // Initial health check
    const performHealthCheck = async () => {
      const status = await checkNIMHealth();
      setHealth(status);
    };

    performHealthCheck();

    // Poll every 5 seconds
    const interval = setInterval(performHealthCheck, 5000);

    // Cleanup on unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="transparent-header">
      <AppBar
        slotLeft={
          <Flex gap="4" align="center">
            <Image src="/logo.png" alt="NVIDIA Logo" width={32} height={32} />
            <Text kind="title/sm">Catalog Enrichment Blueprint</Text>
          </Flex>
        }
        slotRight={
          <HealthIndicators health={health} />
        }
      />
    </div>
  );
}

