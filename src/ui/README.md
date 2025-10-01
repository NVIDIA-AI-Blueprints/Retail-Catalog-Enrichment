# Catalog Enrichment UI

This is the frontend application for the Catalog Enrichment system, built with Next.js, React, and NVIDIA's Kaizen UI (KUI) design system.

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm

### Installation

1. Install dependencies:

```bash
pnpm install
# or
npm install
```

2. Run the development server:

```bash
pnpm dev
# or
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/ui/
├── app/                  # Next.js app directory
│   ├── globals.css      # Global styles with KUI setup
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Home page
├── kui-foundations/     # KUI design system (linked)
├── public/             # Static assets
└── package.json        # Dependencies
```

## KUI Design System

This project uses NVIDIA's Kaizen UI (KUI) design system. Components are imported from `@kui/foundations-react-external`.

### Using KUI Components

```tsx
import { Button, Text, Card } from '@kui/foundations-react-external';

function MyComponent() {
  return (
    <Card>
      <Text kind="title/md">Hello World</Text>
      <Button>Click me</Button>
    </Card>
  );
}
```

### Theme Variables

KUI provides theme-aware CSS variables that automatically support dark mode:

- Background: `bg-surface-base`, `bg-surface-raised`, etc.
- Text: `text-primary`, `text-secondary`, `text-brand`
- Borders: `border-base`, `border-accent-blue`, etc.

## Development

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm start` - Start production server
- `pnpm lint` - Run ESLint

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- KUI Design System (internal NVIDIA resource)








