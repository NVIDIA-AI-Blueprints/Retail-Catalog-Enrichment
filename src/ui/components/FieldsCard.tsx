import { Card, Stack, Text, Flex, FormField, TextInput, TextArea, Tabs, Accordion, Spinner } from '@/kui-foundations-react-external';
import { ProductFields, AugmentedData, PolicyDecision, FAQ } from '@/types';
import { ProcessingSteps } from './ProcessingSteps';

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
        <Stack gap="4" align="center">
          <Spinner size="large" />
          <Text kind="body/regular/md" className="text-secondary">
            Generating FAQs...
          </Text>
        </Stack>
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

interface Props {
  fields: ProductFields;
  augmentedData: AugmentedData | null;
  isAnalyzing: boolean;
  isGenerating: boolean;
  isLoadingFaqs?: boolean;
  onFieldChange: (field: keyof ProductFields, value: string) => void;
}

export function FieldsCard({ fields, augmentedData, isAnalyzing, isGenerating, isLoadingFaqs, onFieldChange }: Props) {
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
                children: "FAQs",
                value: "faqs",
                slotContent: <div style={{ width: '100%' }}><FaqTabContent faqs={augmentedData?.faqs} isLoading={isLoadingFaqs} /></div>
              }
            ]}
          />
        )}
      </Stack>
    </Card>
  );
}
