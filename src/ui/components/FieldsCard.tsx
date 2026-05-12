import { Card, Stack, Text, Flex, FormField, TextInput, TextArea, Tabs, Accordion, Spinner } from '@/kui-foundations-react-external';
import { ProductFields, AugmentedData, PolicyDecision, FAQ, WebInsights, RichProductJson } from '@/types';
import type { ProtocolSchemas } from '@/lib/api';
import { ProcessingSteps } from './ProcessingSteps';
import { JsonObjectViewer, ProtocolsTabContent } from './ProtocolsTabContent';

function PolicyComplianceCard({ decision }: { decision: PolicyDecision }) {
  const isFail = decision.status === 'fail';

  const colors = isFail
    ? {
        border: 'rgba(255, 84, 89, 0.30)',
        bg: 'rgba(255, 84, 89, 0.06)',
        icon: 'rgba(255, 84, 89, 0.12)',
        iconStroke: '#FF5459',
        accent: '#FF5459',
        badgeText: '#FFB4B6',
        badgeBg: 'rgba(255, 84, 89, 0.16)',
        badgeBorder: 'rgba(255, 84, 89, 0.35)',
        mutedText: 'rgba(255, 180, 182, 0.7)',
      }
    : {
        border: 'rgba(118, 185, 0, 0.30)',
        bg: 'rgba(118, 185, 0, 0.06)',
        icon: 'rgba(118, 185, 0, 0.12)',
        iconStroke: '#76B900',
        accent: '#76B900',
        badgeText: '#B8E86B',
        badgeBg: 'rgba(118, 185, 0, 0.16)',
        badgeBorder: 'rgba(118, 185, 0, 0.35)',
        mutedText: 'rgba(184, 232, 107, 0.7)',
      };

  return (
    <div
      style={{
        border: `1px solid ${colors.border}`,
        borderRadius: '16px',
        background: colors.bg,
        padding: '20px',
      }}
    >
      <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
        <div
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '12px',
            background: colors.icon,
            display: 'grid',
            placeItems: 'center',
            flexShrink: 0,
          }}
        >
          {isFail ? (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M8.57 3.22L1.52 14.5a1.67 1.67 0 001.43 2.5h14.1a1.67 1.67 0 001.43-2.5L11.43 3.22a1.67 1.67 0 00-2.86 0z" stroke={colors.iconStroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M10 7.5v3.33M10 14.17h.008" stroke={colors.iconStroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 18.33a8.33 8.33 0 100-16.66 8.33 8.33 0 000 16.66z" stroke={colors.iconStroke} strokeWidth="1.5" />
              <path d="M6.67 10l2.5 2.5 4.16-5" stroke={colors.iconStroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <Flex justify="between" align="start" style={{ marginBottom: '8px' }}>
            <Stack gap="1">
              <Text kind="body/regular/sm" style={{ color: colors.mutedText, letterSpacing: '0.08em', textTransform: 'uppercase', fontSize: '11px' }}>
                Policy Compliance
              </Text>
              <Text kind="body/semibold/md" className="text-primary">
                {decision.label}
              </Text>
            </Stack>
            <span
              style={{
                color: colors.badgeText,
                backgroundColor: colors.badgeBg,
                border: `1px solid ${colors.badgeBorder}`,
                padding: '6px 12px',
                borderRadius: '999px',
                fontSize: '12px',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                flexShrink: 0,
              }}
            >
              {isFail ? 'Does not comply' : 'Complies'}
            </span>
          </Flex>

          <Text kind="body/regular/sm" className="text-subtle" style={{ lineHeight: 1.5 }}>
            {decision.summary}
          </Text>
        </div>
      </div>
    </div>
  );
}

function FaqTabContent({ faqs, isLoading }: { faqs?: FAQ[]; isLoading?: boolean }) {
  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <Spinner size="large" description="Generating FAQs..." />
      </div>
    );
  }

  if (!faqs || faqs.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Text kind="body/regular/md" className="text-secondary">
          No FAQs generated yet. Run analysis to generate product FAQs.
        </Text>
      </div>
    );
  }

  return (
    <div style={{ paddingTop: '16px' }}>
      <Accordion
        multiple
        items={faqs.map((faq, index) => ({
          slotTrigger: (
            <Text kind="body/semibold/md" className="text-primary">
              {faq.question}
            </Text>
          ),
          slotContent: (
            <Text kind="body/regular/md" className="text-primary" style={{ whiteSpace: 'pre-line', lineHeight: 1.6 }}>
              {faq.answer}
            </Text>
          ),
          value: String(index)
        }))}
      />
    </div>
  );
}

function RichProductJsonTabContent({
  value,
  error,
  isLoading,
}: {
  value?: RichProductJson;
  error?: string;
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <Spinner size="large" description="Generating VLM product JSON..." />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Text kind="body/regular/md" className="text-secondary">
          {error}
        </Text>
      </div>
    );
  }

  if (!value) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Text kind="body/regular/md" className="text-secondary">
          No raw data available yet. Run analysis to generate rich product attributes.
        </Text>
      </div>
    );
  }

  return (
    <div style={{ paddingTop: '16px' }}>
      <Stack gap="4">
        <Flex justify="between" align="center">
          <Text kind="body/semibold/md" className="text-primary">
            Rich Product JSON
          </Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Generated by Nemotron VLM
          </Text>
        </Flex>
        <JsonObjectViewer value={value} />
      </Stack>
    </div>
  );
}

type WebInsightsReport = NonNullable<WebInsights['report']>;

const emptyMetric = {
  label: 'Not enough data',
  score: null,
  scale: 'label' as const,
  rationale: '',
};

function splitInsightText(text: string) {
  const clean = text.trim();
  if (!clean) return { title: '', detail: '' };
  const separator = clean.includes(':') ? ':' : clean.includes(' - ') ? ' - ' : '';
  if (separator) {
    const [title, ...rest] = clean.split(separator);
    return { title: title.trim(), detail: rest.join(separator).trim() };
  }
  const words = clean.split(/\s+/);
  return { title: words.slice(0, 5).join(' '), detail: clean };
}

function fallbackReport(insights: WebInsights): WebInsightsReport {
  return {
    executive_summary: insights.summary,
    positioning_tags: [...insights.pros, ...insights.use_cases, ...insights.purchase_considerations]
      .map((item) => splitInsightText(item).title)
      .filter(Boolean)
      .slice(0, 6),
    metrics: {
      customer_sentiment: emptyMetric,
      build_quality: emptyMetric,
      price_segment: emptyMetric,
      retail_confidence: { ...emptyMetric, scale: 'rating_10' },
    },
    retail_insights: [
      ...insights.pros.map((item) => ({ type: 'positive' as const, ...splitInsightText(item) })),
      ...insights.cons.map((item) => ({ type: 'negative' as const, ...splitInsightText(item) })),
    ].filter((item) => item.title || item.detail).slice(0, 8),
    primary_use_cases: insights.use_cases.map(splitInsightText).filter((item) => item.title || item.detail).slice(0, 6),
    customer_sentiment_summary: insights.customer_insights.slice(0, 2).join(' '),
  };
}

function isLowIdentityScope(scope?: WebInsights['research_scope']) {
  return scope === 'category_level' || scope === 'insufficient_identity';
}

function isPlaceholderMetricLabel(label?: string) {
  const normalized = (label || '').trim().toLowerCase();
  return (
    !normalized ||
    normalized === 'qualitative label' ||
    normalized === 'label' ||
    normalized === 'not enough data' ||
    normalized === 'varies by retailer' ||
    normalized === 'varies by retailers' ||
    normalized.startsWith('qualitative ')
  );
}

function formatMetricValue(
  metric: WebInsightsReport['metrics'][keyof WebInsightsReport['metrics']],
  scope?: WebInsights['research_scope'],
  fallbackLabel = 'Unavailable',
) {
  if (isLowIdentityScope(scope)) {
    return isPlaceholderMetricLabel(metric.label)
      ? (scope === 'category_level' ? 'Category signal' : 'Limited')
      : metric.label;
  }
  if (metric.scale === 'percent' && metric.score !== null) {
    return `${Math.round(metric.score)}%`;
  }
  if (metric.scale === 'rating_10' && metric.score !== null) {
    return `${metric.score.toFixed(1).replace('.0', '')}/10`;
  }
  return isPlaceholderMetricLabel(metric.label) ? fallbackLabel : metric.label;
}

function formatScopeLabel(scope?: WebInsights['research_scope']) {
  switch (scope) {
    case 'brand_level':
      return 'Brand-level insights';
    case 'category_level':
      return 'Category-level insights';
    case 'insufficient_identity':
      return 'Limited identity';
    case 'product_specific':
    default:
      return 'Product-specific insights';
  }
}

function formatScopeDetail(insights: WebInsights) {
  if (insights.scope_note) return insights.scope_note;
  switch (insights.research_scope) {
    case 'brand_level':
      return 'A likely brand was identified, but exact model-level evidence was not confirmed.';
    case 'category_level':
      return 'No reliable brand or model was identified, so these insights describe broader category patterns.';
    case 'insufficient_identity':
      return 'The product title is too broad for reliable web research.';
    default:
      return '';
  }
}

function InsightBadge({ tone }: { tone: 'positive' | 'negative' }) {
  const positive = tone === 'positive';
  return (
    <div
      aria-hidden="true"
      style={{
        width: '36px',
        height: '36px',
        borderRadius: '10px',
        display: 'grid',
        placeItems: 'center',
        flexShrink: 0,
        color: positive ? '#9DE64F' : '#FF8A8A',
        background: positive ? 'rgba(118, 185, 0, 0.13)' : 'rgba(255, 84, 89, 0.12)',
        border: positive ? '1px solid rgba(118, 185, 0, 0.28)' : '1px solid rgba(255, 84, 89, 0.25)',
        fontWeight: 800,
      }}
    >
      {positive ? '✓' : '!'}
    </div>
  );
}

function WebInsightsTabContent({
  insights,
  isLoading,
}: {
  insights?: WebInsights;
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <Spinner size="large" description="Researching product insights..." />
      </div>
    );
  }

  if (!insights) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Text kind="body/regular/md" className="text-secondary">
          No web insights available yet. Run analysis to research product insights.
        </Text>
      </div>
    );
  }

  const report = insights.report || fallbackReport(insights);
  const researchScope = insights.research_scope || 'product_specific';
  const showScopeBanner = researchScope !== 'product_specific';
  const hasReportContent = Boolean(
    report.executive_summary ||
    report.customer_sentiment_summary ||
    report.retail_insights.length > 0 ||
    report.primary_use_cases.length > 0
  );
  const emptyStateDetail = insights.warnings[0] || insights.scope_note || insights.summary;
  const panelStyle = {
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: '8px',
    background: 'rgba(18, 22, 31, 0.72)',
    padding: '28px',
  };
  const mutedTextStyle = {
    display: 'block',
    color: '#AAB3C2',
    fontSize: '15px',
    lineHeight: 1.75,
  };
  const metricCards = [
    {
      label: 'Customer Sentiment',
      value: formatMetricValue(report.metrics.customer_sentiment, researchScope, 'Review signal unavailable'),
      rationale: report.metrics.customer_sentiment.rationale,
    },
    {
      label: 'Build Quality',
      value: formatMetricValue(report.metrics.build_quality, researchScope, 'Build signal unavailable'),
      rationale: report.metrics.build_quality.rationale,
    },
    {
      label: 'Price Segment',
      value: formatMetricValue(report.metrics.price_segment, researchScope, 'Price unavailable'),
      rationale: report.metrics.price_segment.rationale,
    },
    {
      label: 'Retail Confidence',
      value: formatMetricValue(report.metrics.retail_confidence, researchScope, 'Limited'),
      rationale: report.metrics.retail_confidence.rationale,
    },
  ];

  return (
    <div style={{ paddingTop: '12px' }}>
      {!hasReportContent ? (
        <div
          style={{
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '8px',
            background: 'rgba(18, 22, 31, 0.72)',
            padding: '28px',
            textAlign: 'left',
          }}
        >
          <Text kind="title/sm" className="text-primary" style={{ display: 'block', marginBottom: '12px', lineHeight: 1.2 }}>
            No Source-Backed Report Yet
          </Text>
          <Text kind="body/regular/md" style={{ display: 'block', color: '#AAB3C2', fontSize: '15px', lineHeight: 1.6 }}>
            {emptyStateDetail || 'Web insights are unavailable for this product.'}
          </Text>
        </div>
      ) : (
        <Stack gap="5">
          {showScopeBanner && (
            <div
              style={{
                border: '1px solid rgba(255, 207, 102, 0.24)',
                borderRadius: '8px',
                background: 'rgba(255, 207, 102, 0.07)',
                padding: '16px 18px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', marginBottom: '6px' }}>
                <span
                  style={{
                    border: '1px solid rgba(255, 207, 102, 0.28)',
                    borderRadius: '999px',
                    background: 'rgba(255, 207, 102, 0.12)',
                    color: '#FFCF66',
                    padding: '7px 11px',
                    fontSize: '13px',
                    lineHeight: 1,
                    fontWeight: 700,
                  }}
                >
                  {formatScopeLabel(researchScope)}
                </span>
                {insights.identity_confidence && (
                  <Text kind="body/semibold/sm" style={{ color: '#D6DBE5', fontSize: '14px', lineHeight: 1 }}>
                    Identity confidence: {insights.identity_confidence}
                  </Text>
                )}
              </div>
              <Text kind="body/regular/md" style={{ ...mutedTextStyle, fontSize: '14px', lineHeight: 1.5 }}>
                {formatScopeDetail(insights)}
              </Text>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
            {metricCards.map((metric) => (
              <div
                key={metric.label}
                title={metric.rationale || metric.label}
                style={{
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: '8px',
                  background: 'rgba(18, 22, 31, 0.72)',
                  padding: '18px',
                  minHeight: '104px',
                }}
              >
                <Text
                  kind="body/semibold/sm"
                  style={{
                    display: 'block',
                    color: '#D6DBE5',
                    fontSize: '14px',
                    lineHeight: 1.25,
                    marginBottom: '12px',
                  }}
                >
                  {metric.label}
                </Text>
                <Text kind="title/sm" className="text-primary" style={{ display: 'block', lineHeight: 1.1 }}>
                  {metric.value}
                </Text>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
            <section style={{ ...panelStyle, minHeight: '280px' }}>
              <Text kind="title/sm" className="text-primary" style={{ display: 'block', marginBottom: '22px', lineHeight: 1.2 }}>
                Executive Summary
              </Text>
              <Text kind="body/regular/md" style={{ ...mutedTextStyle, whiteSpace: 'pre-line' }}>
                {report.executive_summary || insights.summary}
              </Text>
              {report.positioning_tags.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '24px' }}>
                  {report.positioning_tags.map((tag, index) => (
                    <span
                      key={`${tag}-${index}`}
                      style={{
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '999px',
                        background: 'rgba(255,255,255,0.045)',
                        color: '#D6DBE5',
                        padding: '8px 14px',
                        fontSize: '13px',
                        lineHeight: 1,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </section>

            {report.retail_insights.length > 0 && (
              <section style={{ ...panelStyle, minHeight: '280px' }}>
                <Text kind="title/sm" className="text-primary" style={{ display: 'block', marginBottom: '22px', lineHeight: 1.2 }}>
                  Retail Insights
                </Text>
                <Stack gap="4">
                  {report.retail_insights.map((item, index) => (
                    <div
                      key={`${item.title}-${index}`}
                      style={{
                        display: 'flex',
                        gap: '16px',
                        alignItems: 'flex-start',
                        border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: '8px',
                        background: 'rgba(255,255,255,0.035)',
                        padding: '18px',
                      }}
                    >
                      <InsightBadge tone={item.type} />
                      <div style={{ minWidth: 0 }}>
                        <Text kind="body/semibold/md" className="text-primary" style={{ display: 'block', marginBottom: '6px', lineHeight: 1.25 }}>
                          {item.title}
                        </Text>
                        {item.detail && (
                          <Text kind="body/regular/md" style={{ ...mutedTextStyle, lineHeight: 1.55 }}>
                            {item.detail}
                          </Text>
                        )}
                      </div>
                    </div>
                  ))}
                </Stack>
              </section>
            )}

            {report.primary_use_cases.length > 0 && (
              <section style={{ ...panelStyle, minHeight: '240px' }}>
                <Text kind="title/sm" className="text-primary" style={{ display: 'block', marginBottom: '22px', lineHeight: 1.2 }}>
                  Primary Use Cases
                </Text>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '14px' }}>
                  {report.primary_use_cases.map((item, index) => (
                    <div
                      key={`${item.title}-${index}`}
                      style={{
                        display: 'flex',
                        gap: '14px',
                        alignItems: 'flex-start',
                        border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: '8px',
                        background: 'rgba(255,255,255,0.035)',
                        padding: '16px',
                      }}
                    >
                      <span
                        aria-hidden="true"
                        style={{
                          width: '36px',
                          height: '36px',
                          borderRadius: '10px',
                          display: 'grid',
                          placeItems: 'center',
                          flexShrink: 0,
                          color: '#9DE64F',
                          background: 'rgba(118, 185, 0, 0.13)',
                          border: '1px solid rgba(118, 185, 0, 0.28)',
                          fontSize: '13px',
                          fontWeight: 800,
                        }}
                      >
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      <div style={{ minWidth: 0 }}>
                        <Text kind="body/semibold/md" className="text-primary" style={{ display: 'block', marginBottom: '6px', lineHeight: 1.25 }}>
                          {item.title}
                        </Text>
                        {item.detail && (
                          <Text kind="body/regular/md" style={{ ...mutedTextStyle, lineHeight: 1.55 }}>
                            {item.detail}
                          </Text>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section style={{ ...panelStyle, minHeight: '240px' }}>
              <Text kind="title/sm" className="text-primary" style={{ display: 'block', marginBottom: '22px', lineHeight: 1.2 }}>
                Customer Sentiment
              </Text>
              <Text kind="body/regular/md" style={{ ...mutedTextStyle, whiteSpace: 'pre-line' }}>
                {report.customer_sentiment_summary || 'Not enough source-backed customer sentiment was found for this product.'}
              </Text>
            </section>
          </div>
        </Stack>
      )}
    </div>
  );
}

interface Props {
  fields: ProductFields;
  augmentedData: AugmentedData | null;
  isAnalyzing: boolean;
  isGenerating: boolean;
  isLoadingFaqs?: boolean;
  isLoadingRichProductJson?: boolean;
  isLoadingWebInsights?: boolean;
  protocolSchemas?: ProtocolSchemas | null;
  isLoadingProtocols?: boolean;
  onFieldChange: (field: keyof ProductFields, value: string) => void;
}

export function FieldsCard({ fields, augmentedData, isAnalyzing, isGenerating, isLoadingFaqs, isLoadingRichProductJson, isLoadingWebInsights, protocolSchemas, isLoadingProtocols, onFieldChange }: Props) {
  const disabled = isAnalyzing || isGenerating;

  const detailsContent = (
    <Stack gap="4">
      {augmentedData?.policyDecision && (
        <PolicyComplianceCard decision={augmentedData.policyDecision} />
      )}

      <div>
        <FormField slotLabel="Title">
          {(args: any) => (
            <TextInput
              {...args}
              placeholder=""
              size="medium"
              value={fields.title}
              onChange={(e: any) => onFieldChange('title', e.target.value)}
              disabled={disabled}
            />
          )}
        </FormField>
        {augmentedData && (
          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
            <Stack gap="2">
              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
              <Text kind="body/regular/md" className="text-primary">{augmentedData.title}</Text>
            </Stack>
          </div>
        )}
      </div>

      <div>
        <FormField slotLabel="Description">
          {(args: any) => (
            <TextArea
              {...args}
              placeholder=""
              size="medium"
              resizeable="manual"
              value={fields.description}
              onChange={(e: any) => onFieldChange('description', e.target.value)}
              disabled={disabled}
              attributes={{
                TextAreaElement: { rows: 3 }
              }}
            />
          )}
        </FormField>
        {augmentedData && (
          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
            <Stack gap="2">
              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
              <Text kind="body/regular/md" className="text-primary" style={{ whiteSpace: 'pre-line' }}>
                {augmentedData.description}
              </Text>
            </Stack>
          </div>
        )}
      </div>

      <div>
        <FormField slotLabel="Colors">
          {(args: any) => (
            <TextInput
              {...args}
              placeholder=""
              size="medium"
              value={fields.color}
              onChange={(e: any) => onFieldChange('color', e.target.value)}
              disabled={disabled}
            />
          )}
        </FormField>
        {augmentedData && augmentedData.colors.length > 0 && (
          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
            <Stack gap="2">
              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
              <Text kind="body/regular/md" className="text-primary">{augmentedData.colors.join(', ')}</Text>
            </Stack>
          </div>
        )}
      </div>

      <div>
        <FormField slotLabel="Categories">
          {(args: any) => (
            <TextInput
              {...args}
              placeholder=""
              size="medium"
              value={fields.categories}
              onChange={(e: any) => onFieldChange('categories', e.target.value)}
              disabled={disabled}
            />
          )}
        </FormField>
        {augmentedData?.categories && augmentedData.categories.length > 0 && (
          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
            <Stack gap="2">
              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
              <Text kind="body/regular/md" className="text-primary">{augmentedData.categories.join(', ')}</Text>
            </Stack>
          </div>
        )}
      </div>

      <div>
        <FormField slotLabel="Tags">
          {(args: any) => (
            <TextInput
              {...args}
              placeholder=""
              size="medium"
              value={fields.tags}
              onChange={(e: any) => onFieldChange('tags', e.target.value)}
              disabled={disabled}
            />
          )}
        </FormField>
        {augmentedData && augmentedData.tags.length > 0 && (
          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
            <Stack gap="2">
              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
              <Text kind="body/regular/md" className="text-primary">{augmentedData.tags.join(', ')}</Text>
            </Stack>
          </div>
        )}
      </div>
    </Stack>
  );

  return (
    <Card>
      <Stack gap="6">
        <Text kind="title/md" className="text-primary">Fields</Text>

        {isAnalyzing ? (
          <div>
            <ProcessingSteps isAnalyzing={isAnalyzing} hasAugmentedData={!!augmentedData} />
          </div>
        ) : (
          <Tabs
            kind="secondary"
            items={[
              {
                children: "Details",
                value: "details",
                slotContent: <div style={{ width: '100%' }}>{detailsContent}</div>
              },
              {
                children: "Raw data",
                value: "vlm-json",
                slotContent: <div style={{ width: '100%' }}><RichProductJsonTabContent value={augmentedData?.richProductJson} error={augmentedData?.richProductJsonError} isLoading={isLoadingRichProductJson} /></div>
              },
              {
                children: "FAQs",
                value: "faqs",
                slotContent: <div style={{ width: '100%' }}><FaqTabContent faqs={augmentedData?.faqs} isLoading={isLoadingFaqs} /></div>
              },
              {
                children: "Web Insights",
                value: "web-insights",
                slotContent: <div style={{ width: '100%' }}><WebInsightsTabContent insights={augmentedData?.webInsights} isLoading={isLoadingWebInsights} /></div>
              },
              {
                children: "Protocols",
                value: "protocols",
                slotContent: <div style={{ width: '100%' }}><ProtocolsTabContent protocolSchemas={protocolSchemas} isLoading={isLoadingProtocols} /></div>
              }
            ]}
          />
        )}
      </Stack>
    </Card>
  );
}
