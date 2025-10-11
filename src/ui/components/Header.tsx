import { AppBar, Text, Button, Flex } from '@/kui-foundations-react-external';
import Image from 'next/image';

export function Header() {
  return (
    <AppBar
      slotLeft={
        <Flex gap="4" align="center">
          <Image src="/logo.png" alt="NVIDIA Logo" width={32} height={32} />
          <Text kind="title/sm">Catalog Enrichment Blueprint</Text>
        </Flex>
      }
      slotRight={
        <Flex gap="3" align="center">
          <Button kind="tertiary" size="small">Documentation</Button>
          <Button kind="tertiary" size="small">Settings</Button>
        </Flex>
      }
    />
  );
}

