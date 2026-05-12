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

export type RichProductJson = Record<string, unknown>;

export interface WebInsightSource {
  title: string;
  url: string;
  published_date: string | null;
  snippet: string;
}

export interface WebInsightMetric {
  label: string;
  score: number | null;
  scale: 'percent' | 'rating_10' | 'label';
  rationale: string;
}

export interface RetailInsight {
  type: 'positive' | 'negative';
  title: string;
  detail: string;
}

export interface InsightUseCase {
  title: string;
  detail: string;
}

export type WebInsightsResearchScope = 'product_specific' | 'brand_level' | 'category_level' | 'insufficient_identity';

export type WebInsightsIdentityConfidence = 'high' | 'medium' | 'low' | 'none';

export interface WebInsightsReport {
  executive_summary: string;
  positioning_tags: string[];
  metrics: {
    customer_sentiment: WebInsightMetric;
    build_quality: WebInsightMetric;
    price_segment: WebInsightMetric;
    retail_confidence: WebInsightMetric;
  };
  retail_insights: RetailInsight[];
  primary_use_cases: InsightUseCase[];
  customer_sentiment_summary: string;
}

export interface WebInsights {
  summary: string;
  pros: string[];
  cons: string[];
  use_cases: string[];
  customer_insights: string[];
  purchase_considerations: string[];
  search_queries: string[];
  sources: WebInsightSource[];
  warnings: string[];
  locale: string;
  research_scope?: WebInsightsResearchScope;
  identity_confidence?: WebInsightsIdentityConfidence;
  detected_brand?: string | null;
  detected_model?: string | null;
  scope_note?: string;
  identity_evidence?: string[];
  report?: WebInsightsReport;
}

export type ManualKnowledge = Record<string, string>;

export interface ManualExtractResult {
  filename: string;
  chunk_count: number;
  knowledge: ManualKnowledge;
}

export interface AugmentedData {
  title: string;
  description: string;
  colors: string[];
  tags: string[];
  categories?: string[];
  policyDecision?: PolicyDecision;
  richProductJson?: RichProductJson;
  richProductJsonError?: string;
  faqs?: FAQ[];
  webInsights?: WebInsights;
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
