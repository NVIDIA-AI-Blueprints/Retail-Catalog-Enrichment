import { useState, useRef, useEffect, useMemo, type ReactNode } from 'react';
import { Stack, Text, Flex, SegmentedControl, Spinner } from '@/kui-foundations-react-external';
import type { ProtocolSchemas } from '@/lib/api';

const TOKEN_COLORS = {
  key: '#9CDCFE',       // light blue — keys
  string: '#CE9178',    // warm orange — string values
  number: '#B5CEA8',    // soft green — numbers
  boolean: '#569CD6',   // blue — true/false
  null: '#6A737D',      // gray — null
  bracket: '#D4D4D4',   // light gray — {}[]
  punctuation: '#808080', // gray — : ,
} as const;

function highlightJson(json: string): ReactNode[] {
  const tokenRegex = /("(?:\\.|[^"\\])*")\s*(:)|("(?:\\.|[^"\\])*")|([-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(\btrue\b|\bfalse\b)|(\bnull\b)|([{}[\],])/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  while ((match = tokenRegex.exec(json)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(json.slice(lastIndex, match.index));
    }

    if (match[1] && match[2]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.key }}>{match[1]}</span>);
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.punctuation }}>{match[2]}</span>);
    } else if (match[3]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.string }}>{match[3]}</span>);
    } else if (match[4]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.number }}>{match[4]}</span>);
    } else if (match[5]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.boolean }}>{match[5]}</span>);
    } else if (match[6]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.null }}>{match[6]}</span>);
    } else if (match[7]) {
      nodes.push(<span key={i++} style={{ color: TOKEN_COLORS.bracket }}>{match[7]}</span>);
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < json.length) {
    nodes.push(json.slice(lastIndex));
  }

  return nodes;
}

interface ProtocolsTabContentProps {
  protocolSchemas?: ProtocolSchemas | null;
  isLoading?: boolean;
}

export function ProtocolsTabContent({ protocolSchemas, isLoading }: ProtocolsTabContentProps) {
  const [activeProtocol, setActiveProtocol] = useState<string>('acp');
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout>>(null);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const jsonString = useMemo(() => {
    if (!protocolSchemas) return '';
    const schema = activeProtocol === 'acp' ? protocolSchemas.acp : protocolSchemas.ucp;
    return JSON.stringify(schema, null, 2);
  }, [protocolSchemas, activeProtocol]);

  const highlightedJson = useMemo(() => {
    if (!jsonString) return null;
    return highlightJson(jsonString);
  }, [jsonString]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API requires secure context
    }
  };

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <Spinner size="large" description="Generating protocol schemas..." />
      </div>
    );
  }

  if (!protocolSchemas) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Text kind="body/regular/md" className="text-secondary">
          No protocol data available yet. Run analysis to generate protocol schemas.
        </Text>
      </div>
    );
  }

  return (
    <div style={{ paddingTop: '16px' }}>
      <Stack gap="4">
        <Flex justify="between" align="center">
          <SegmentedControl
            size="small"
            value={activeProtocol}
            onValueChange={setActiveProtocol}
            items={[
              { value: 'acp', children: 'ACP' },
              { value: 'ucp', children: 'UCP' },
            ]}
          />
          <Text kind="body/regular/sm" className="text-secondary">
            Generated Schema
          </Text>
        </Flex>

        <div
          style={{
            position: 'relative',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '12px',
            padding: '16px',
            maxHeight: '400px',
            overflowY: 'auto',
          }}
          className="border border-base"
        >
          <button
            onClick={handleCopy}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              color: copied ? '#76B900' : 'rgba(255, 255, 255, 0.5)',
              padding: '4px 10px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
              transition: 'color 0.15s, border-color 0.15s',
              borderColor: copied ? 'rgba(118, 185, 0, 0.4)' : 'rgba(255, 255, 255, 0.12)',
            }}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <pre
            className="text-primary"
            style={{
              fontFamily: 'monospace',
              fontSize: '13px',
              lineHeight: 1.6,
              margin: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {highlightedJson}
          </pre>
        </div>
      </Stack>
    </div>
  );
}
