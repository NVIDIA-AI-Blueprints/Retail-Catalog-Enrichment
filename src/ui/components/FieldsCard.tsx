import { Card, Stack, Text, FormField, TextInput, TextArea, Spinner } from '@/kui-foundations-react-external';
import { ProductFields, AugmentedData } from '@/types';

interface Props {
  fields: ProductFields;
  augmentedData: AugmentedData | null;
  isAnalyzing: boolean;
  isGenerating: boolean;
  onFieldChange: (field: keyof ProductFields, value: string) => void;
}

export function FieldsCard({ fields, augmentedData, isAnalyzing, isGenerating, onFieldChange }: Props) {
  const disabled = isAnalyzing || isGenerating;

  return (
    <Card>
      <Stack gap="6">
        <Text kind="title/md" className="text-primary">Fields</Text>
        
        {isAnalyzing ? (
          <div className="flex items-center justify-center py-16">
            <Stack gap="4" align="center">
              <Spinner size="large" aria-label="Analyzing image with VLM..." />
            </Stack>
          </div>
        ) : (
          <Stack gap="4">
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
        )}
      </Stack>
    </Card>
  );
}

