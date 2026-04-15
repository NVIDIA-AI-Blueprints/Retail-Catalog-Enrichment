export interface ProductFields {
  title: string;
  description: string;
  color: string;
  categories: string;
  tags: string;
  price: string;
  brandInstructions: string;
}

export interface PolicyMatch {
  document_name: string;
  policy_title: string;
  rule_title: string;
  reason: string;
  evidence: string[];
}

export interface PolicyDecision {
  status: 'pass' | 'fail';
  label: string;
  summary: string;
  matched_policies: PolicyMatch[];
  warnings: string[];
  evidence_note: string;
}

export interface PolicyDocument {
  document_hash: string;
  filename: string;
  file_size: number;
  chunk_count: number;
  created_at: number;
  updated_at: number;
}

export interface PolicyUploadResult {
  document_hash: string;
  filename: string;
  chunk_count: number;
  already_loaded: boolean;
  processed: boolean;
}

export interface FAQ {
  question: string;
  answer: string;
}

export interface AugmentedData {
  title: string;
  description: string;
  colors: string[];
  tags: string[];
  categories?: string[];
  policyDecision?: PolicyDecision;
  faqs?: FAQ[];
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
