export interface ProductFields {
  title: string;
  description: string;
  color: string;
  categories: string;
  tags: string;
  price: string;
  brandInstructions: string;
}

export interface AugmentedData {
  title: string;
  description: string;
  colors: string[];
  tags: string[];
  categories?: string[];
}

export interface ImageMetadata {
  name: string;
  size: string;
  dimensions?: string;
}

export interface LocaleOption {
  value: string;
  children: string;
}

export const SUPPORTED_LOCALES: LocaleOption[] = [
  { value: 'en-US', children: 'English (US)' },
  { value: 'en-GB', children: 'English (UK)' },
  { value: 'en-AU', children: 'English (Australia)' },
  { value: 'en-CA', children: 'English (Canada)' },
  { value: 'es-ES', children: 'Spanish (Spain)' },
  { value: 'es-MX', children: 'Spanish (Mexico)' },
  { value: 'es-AR', children: 'Spanish (Argentina)' },
  { value: 'es-CO', children: 'Spanish (Colombia)' },
  { value: 'fr-FR', children: 'French (France)' },
  { value: 'fr-CA', children: 'French (Canada)' }
];

export type HealthState = "healthy" | "unhealthy" | "checking";

export interface NIMHealthStatus {
  vlm: HealthState;
  llm: HealthState;
  flux: HealthState;
  trellis: HealthState;
}

