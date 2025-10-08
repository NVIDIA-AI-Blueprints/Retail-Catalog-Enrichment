'use client';

import { 
  AppBar, 
  Text, 
  Card, 
  Stack, 
  Button,
  Flex,
  FormField,
  TextInput,
  TextArea,
  Spinner
} from '@/kui-foundations-react-external';
import Image from 'next/image';
import { useState, useRef } from 'react';

interface ProductFields {
  title: string;
  description: string;
  color: string;
  categories: string;
  tags: string;
  price: string;
}

interface AugmentedData {
  title: string;
  description: string;
  colors: string[];
  tags: string[];
  categories?: string[];
}

function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [imageMetadata, setImageMetadata] = useState<{
    name: string;
    size: string;
    dimensions?: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fields, setFields] = useState<ProductFields>({
    title: '',
    description: '',
    color: '',
    categories: '',
    tags: '',
    price: ''
  });
  const [augmentedData, setAugmentedData] = useState<AugmentedData | null>(null);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);

  const formatFileSize = (bytes: number): string => 
    bytes < 1024 ? `${bytes} bytes` : 
    bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : 
    `${(bytes / 1048576).toFixed(1)} MB`;

  const handleFileUpload = async (file: File) => {
    if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
      alert('Please upload a PNG, JPG, or JPEG file');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setIsUploading(true);
    setUploadedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new window.Image();
      img.onload = () => {
        setImageMetadata({
          name: file.name,
          size: formatFileSize(file.size),
          dimensions: `${img.width} × ${img.height}`
        });
        setIsUploading(false);
      };
      img.src = e.target?.result as string;
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleRemoveImage = () => {
    setUploadedImage(null);
    setUploadedFile(null);
    setImageMetadata(null);
    setAugmentedData(null);
    setGeneratedImageUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleGenerate = async () => {
    if (!uploadedFile) return;

    setIsGenerating(true);
    
    try {
      const formData = new FormData();
      formData.append('image', uploadedFile);
      formData.append('locale', 'en-US');

      const hasFields = Object.values(fields).some(val => val.trim() !== '');
      if (hasFields) {
        const productData: any = {};
        if (fields.title) productData.title = fields.title;
        if (fields.description) productData.description = fields.description;
        if (fields.categories) productData.categories = fields.categories.split(',').map(c => c.trim());
        if (fields.tags) productData.tags = fields.tags.split(',').map(t => t.trim());
        if (fields.price) productData.price = parseFloat(fields.price);
        formData.append('product_data', JSON.stringify(productData));
      }

      const response = await fetch('http://localhost:8000/vlm/describe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to generate enriched data');
      }

      const data = await response.json();
      setAugmentedData({
        title: data.title || '',
        description: data.description || '',
        colors: data.colors || [],
        tags: data.tags || [],
        categories: data.categories || []
      });

      if (data.generated_image_b64) {
        setGeneratedImageUrl(`data:image/png;base64,${data.generated_image_b64}`);
      }
    } catch (error) {
      console.error('Generation error:', error);
      alert(error instanceof Error ? error.message : 'Failed to generate enriched data');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-base">
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

      <main className="pt-16 bg-surface-base min-h-screen">
        <div className="max-w-7xl mx-auto px-8" style={{ paddingTop: '48px', paddingBottom: '48px' }}>
          <Stack gap="8">
            <div style={{ padding: '0 16px' }}>
              <Button 
                kind="primary" 
                size="large" 
                className="nvidia-green-button"
                disabled={!uploadedImage || isGenerating}
                onClick={handleGenerate}
              >
                {isGenerating ? (
                  <Flex gap="2" align="center">
                    <Spinner size="small" />
                    <span>Generating...</span>
                  </Flex>
                ) : uploadedImage ? 'Generate Enriched Data' : 'Upload Image to Start'}
              </Button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', padding: '0 16px' }}>
              <Card>
                <Stack gap="6">
                  <Flex justify="between" align="center">
                    <Text kind="title/md" className="text-primary">Image</Text>
                    {uploadedImage && (
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-300" style={{ backgroundColor: 'var(--color-green-300)' }}></div>
                        <Text kind="body/regular/sm" className="text-subtle">Ready</Text>
                      </div>
                    )}
                  </Flex>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload(file);
                    }}
                    className="hidden"
                  />
                  
                  {isUploading ? (
                    <div className="border-2 border-dashed border-base rounded-lg p-16 text-center bg-surface-sunken">
                      <Stack gap="4" align="center">
                        <Spinner size="large" description="Uploading image..." />
                      </Stack>
                    </div>
                  ) : uploadedImage ? (
                    <>
                      <div 
                        className="relative rounded-lg overflow-hidden nvidia-green-border" 
                        style={{ 
                          minHeight: '400px',
                          backgroundColor: 'var(--color-gray-1000)',
                          borderWidth: '2px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <img 
                          src={uploadedImage} 
                          alt="Uploaded preview" 
                          style={{ 
                            maxWidth: '100%', 
                            maxHeight: '400px',
                            width: 'auto',
                            height: 'auto',
                            objectFit: 'contain',
                            display: 'block'
                          }}
                        />
                        <div 
                          style={{
                            position: 'absolute',
                            bottom: '12px',
                            left: '0',
                            right: '0',
                            display: 'flex',
                            gap: '8px',
                            justifyContent: 'center',
                            alignItems: 'center'
                          }}
                        >
                          <Button kind="secondary" size="small" onClick={() => fileInputRef.current?.click()}>
                            Change
                          </Button>
                          <Button kind="secondary" size="small" onClick={handleRemoveImage}>
                            Remove
                          </Button>
                        </div>
                      </div>
                      
                      {imageMetadata && (
                        <div className="bg-surface-sunken rounded-lg p-4 border border-base">
                          <Stack gap="2">
                            <Flex justify="between" align="center">
                              <Text kind="body/regular/sm" className="text-subtle">File Name</Text>
                              <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.name}</Text>
                            </Flex>
                            <Flex justify="between" align="center">
                              <Text kind="body/regular/sm" className="text-subtle">Size</Text>
                              <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.size}</Text>
                            </Flex>
                            {imageMetadata.dimensions && (
                              <Flex justify="between" align="center">
                                <Text kind="body/regular/sm" className="text-subtle">Dimensions</Text>
                                <Text kind="body/semibold/sm" className="text-primary">{imageMetadata.dimensions}</Text>
                              </Flex>
                            )}
                          </Stack>
                        </div>
                      )}
                    </>
                  ) : (
                    <div 
                      className="border-2 border-dashed border-base rounded-lg p-16 text-center bg-surface-sunken hover:bg-surface-raised transition-colors cursor-pointer"
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                    >
                      <Stack gap="4" align="center">
                        <div className="w-20 h-20 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                          <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                        </div>
                        <Stack gap="2" align="center">
                          <Text kind="body/semibold/md" className="text-primary">
                            Click to upload or drag and drop
                          </Text>
                          <Text kind="body/regular/sm" className="text-subtle">
                            PNG, JPG or JPEG (max. 10MB)
                          </Text>
                        </Stack>
                      </Stack>
                    </div>
                  )}
                </Stack>
              </Card>

              <Card>
                <Stack gap="6">
                  <Text kind="title/md" className="text-primary">Fields</Text>
                  
                  {isGenerating ? (
                    <div className="flex items-center justify-center py-16">
                      <Stack gap="4" align="center">
                        <Spinner size="large" description="Analyzing image and generating enriched data..." />
                      </Stack>
                    </div>
                  ) : (
                    <Stack gap="4">
                      <div>
                        <FormField slotLabel="Title">
                          {(args: any) => (
                            <TextInput 
                              {...args}
                              placeholder="Enter product title"
                              size="medium"
                              value={fields.title}
                              onChange={(e: any) => setFields(prev => ({ ...prev, title: e.target.value }))}
                              disabled={isGenerating}
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
                              placeholder="Enter product description"
                              size="medium"
                              resizeable="manual"
                              value={fields.description}
                              onChange={(e: any) => setFields(prev => ({ ...prev, description: e.target.value }))}
                              disabled={isGenerating}
                              attributes={{
                                TextAreaElement: {
                                  rows: 3
                                }
                              }}
                            />
                          )}
                        </FormField>
                        {augmentedData && (
                          <div className="mt-2 p-3 rounded-lg border border-base bg-surface-sunken">
                            <Stack gap="2">
                              <Text kind="body/semibold/md" className="nvidia-green-text">Augmented:</Text>
                              <Text kind="body/regular/md" className="text-primary">{augmentedData.description}</Text>
                            </Stack>
                          </div>
                        )}
                      </div>

                      <div>
                        <FormField slotLabel="Colors">
                          {(args: any) => (
                            <TextInput 
                              {...args}
                              placeholder="e.g., Black, Red, Blue"
                              size="medium"
                              value={fields.color}
                              onChange={(e: any) => setFields(prev => ({ ...prev, color: e.target.value }))}
                              disabled={isGenerating}
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
                              placeholder="e.g., furniture, electronics, clothing"
                              size="medium"
                              value={fields.categories}
                              onChange={(e: any) => setFields(prev => ({ ...prev, categories: e.target.value }))}
                              disabled={isGenerating}
                            />
                          )}
                        </FormField>
                        {augmentedData && augmentedData.categories && augmentedData.categories.length > 0 && (
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
                              placeholder="e.g., fashion, accessories, luxury"
                              size="medium"
                              value={fields.tags}
                              onChange={(e: any) => setFields(prev => ({ ...prev, tags: e.target.value }))}
                              disabled={isGenerating}
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

                      <FormField slotLabel="Price">
                        {(args: any) => (
                          <TextInput 
                            {...args}
                            type="number"
                            placeholder="0.00"
                            size="medium"
                            value={fields.price}
                            onChange={(e: any) => setFields(prev => ({ ...prev, price: e.target.value }))}
                            disabled={isGenerating}
                          />
                        )}
                      </FormField>
                    </Stack>
                  )}
                </Stack>
              </Card>
            </div>

            <div style={{ padding: '0 16px' }}>
              <Stack gap="6">
                <Text kind="title/lg" className="text-primary">Generated Image Variations</Text>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
                  <Card>
                    <Stack gap="4">
                      {generatedImageUrl ? (
                        <div 
                          className="rounded-lg overflow-hidden"
                          style={{ 
                            minHeight: '300px',
                            backgroundColor: 'var(--color-gray-1000)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                        >
                          <img 
                            src={generatedImageUrl} 
                            alt="Generated variation" 
                            style={{ 
                              maxWidth: '100%', 
                              maxHeight: '300px',
                              width: 'auto',
                              height: 'auto',
                              objectFit: 'contain',
                              display: 'block'
                            }}
                          />
                        </div>
                      ) : (
                        <div 
                          className="bg-surface-sunken rounded-lg border-2 border-dashed border-base"
                          style={{ 
                            minHeight: '300px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                        >
                          {isGenerating ? (
                            <Stack gap="3" align="center">
                              <Spinner size="large" description="Generating image..." />
                            </Stack>
                          ) : (
                            <Stack gap="3" align="center">
                              <div className="w-16 h-16 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                              </div>
                              <Text kind="body/regular/sm" className="text-subtle">Variation 1</Text>
                            </Stack>
                          )}
                        </div>
                      )}
                      <Button 
                        kind="secondary" 
                        size="medium" 
                        disabled={!generatedImageUrl}
                        onClick={() => {
                          const link = document.createElement('a');
                          link.href = generatedImageUrl!;
                          link.download = 'generated-variation-1.png';
                          link.click();
                        }}
                      >
                        Download
                      </Button>
                    </Stack>
                  </Card>

                  <Card>
                    <Stack gap="4">
                      <div 
                        className="bg-surface-sunken rounded-lg border-2 border-dashed border-base"
                        style={{ 
                          minHeight: '300px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <Stack gap="3" align="center">
                          <div className="w-16 h-16 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <Text kind="body/regular/sm" className="text-subtle">Variation 2</Text>
                        </Stack>
                      </div>
                      <Button kind="secondary" size="medium" disabled>
                        Download
                      </Button>
                    </Stack>
                  </Card>

                  <Card>
                    <Stack gap="4">
                      <div 
                        className="bg-surface-sunken rounded-lg border-2 border-dashed border-base"
                        style={{ 
                          minHeight: '300px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <Stack gap="3" align="center">
                          <div className="w-16 h-16 bg-surface-raised rounded-lg flex items-center justify-center border border-base">
                            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="var(--text-color-subtle)">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <Text kind="body/regular/sm" className="text-subtle">Variation 3</Text>
                        </Stack>
                      </div>
                      <Button kind="secondary" size="medium" disabled>
                        Download
                      </Button>
                    </Stack>
                  </Card>
                </div>
              </Stack>
            </div>
          </Stack>
        </div>
      </main>
    </div>
  );
}

export default Home;
