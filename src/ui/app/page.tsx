'use client';

import { Stack, Button, Flex, FormField, Select, Spinner } from '@/kui-foundations-react-external';
import { useState, useRef } from 'react';
import { Header } from '@/components/Header';
import { ImageUploadCard } from '@/components/ImageUploadCard';
import { FieldsCard } from '@/components/FieldsCard';
import { GeneratedVariationsSection } from '@/components/GeneratedVariationsSection';
import { ProductFields, AugmentedData, ImageMetadata, SUPPORTED_LOCALES } from '@/types';
import { analyzeImage, generateImageVariation, generate3DModel, prepareProductData } from '@/lib/api';
import { formatFileSize } from '@/lib/utils';

function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzingFields, setIsAnalyzingFields] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [imageMetadata, setImageMetadata] = useState<ImageMetadata | null>(null);
  const [locale, setLocale] = useState<string>('en-US');
  const [fields, setFields] = useState<ProductFields>({
    title: '',
    description: '',
    color: '',
    categories: '',
    tags: '',
    price: ''
  });
  const [augmentedData, setAugmentedData] = useState<AugmentedData | null>(null);
  const [generatedImages, setGeneratedImages] = useState<(string | null)[]>([null, null]);
  const [generated3DModel, setGenerated3DModel] = useState<string | null>(null);
  const [model3DError, setModel3DError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleReset = () => {
    setUploadedImage(null);
    setUploadedFile(null);
    setImageMetadata(null);
    setAugmentedData(null);
    setGeneratedImages([null, null]);
    setGenerated3DModel(null);
    setModel3DError(null);
    setLocale('en-US');
    setFields({ title: '', description: '', color: '', categories: '', tags: '', price: '' });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleGenerate = async () => {
    if (!uploadedFile) return;

    setAugmentedData(null);
    setGeneratedImages([null, null]);
    setGenerated3DModel(null);
    setModel3DError(null);
    
    try {
      setIsAnalyzingFields(true);
      const productData = prepareProductData(fields);
      
      const analyzeData = await analyzeImage({ file: uploadedFile, locale, productData });
      
      setAugmentedData({
        title: analyzeData.title || '',
        description: analyzeData.description || '',
        colors: analyzeData.colors || [],
        tags: analyzeData.tags || [],
        categories: analyzeData.categories || []
      });
      setIsAnalyzingFields(false);

      setIsGeneratingImage(true);
      
      const variationParams = {
        file: uploadedFile,
        locale,
        title: analyzeData.title || '',
        description: analyzeData.description || '',
        categories: analyzeData.categories || [],
        tags: analyzeData.tags || [],
        colors: analyzeData.colors || [],
        enhancedProduct: (analyzeData as any).enhanced_product
      };

      await Promise.all([
        (async () => {
          try {
            const imageUrl = await generateImageVariation(variationParams);
            setGeneratedImages(prev => [imageUrl, prev[1]]);
          } catch (error) {
            console.error('Error generating variation 1:', error);
          }
        })(),
        (async () => {
          try {
            const imageUrl = await generateImageVariation(variationParams);
            setGeneratedImages(prev => [prev[0], imageUrl]);
          } catch (error) {
            console.error('Error generating variation 2:', error);
          }
        })(),
        (async () => {
          try {
            const modelUrl = await generate3DModel(uploadedFile);
            setGenerated3DModel(modelUrl);
            if (!modelUrl) {
              setModel3DError('3D model generation completed but no model data was returned');
            }
          } catch (error) {
            console.error('Error generating 3D model:', error);
            setModel3DError(error instanceof Error ? error.message : 'Failed to generate 3D model');
          }
        })()
      ]);
    } catch (error) {
      console.error('Generation error:', error);
      alert(error instanceof Error ? error.message : 'Failed to generate enriched data');
    } finally {
      setIsAnalyzingFields(false);
      setIsGeneratingImage(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-base">
      <Header />
      
      <main className="pt-16 bg-surface-base min-h-screen">
        <div className="max-w-7xl mx-auto px-8" style={{ paddingTop: '48px', paddingBottom: '48px' }}>
          <Stack gap="8">
            <div style={{ padding: '0 16px' }}>
              <Flex gap="4" align="center">
                <Button 
                  kind="primary" 
                  size="large" 
                  className="nvidia-green-button"
                  disabled={!uploadedImage || isAnalyzingFields || isGeneratingImage}
                  onClick={handleGenerate}
                >
                  {isAnalyzingFields ? (
                    <Flex gap="2" align="center">
                      <Spinner size="small" aria-label="Analyzing" />
                      <span>Analyzing...</span>
                    </Flex>
                  ) : isGeneratingImage ? (
                    <Flex gap="2" align="center">
                      <Spinner size="small" aria-label="Generating" />
                      <span>Generating Image...</span>
                    </Flex>
                  ) : uploadedImage ? 'Generate Enriched Data' : 'Upload Image to Start'}
                </Button>
                
                <div style={{ width: '240px' }}>
                  <FormField slotLabel="Locale">
                    {(args: any) => (
                      <Select
                        {...args}
                        items={SUPPORTED_LOCALES}
                        value={locale}
                        onValueChange={setLocale}
                        placeholder="Select locale"
                        size="large"
                        disabled={isAnalyzingFields || isGeneratingImage}
                      />
                    )}
                  </FormField>
                </div>

                <Button 
                  kind="secondary" 
                  size="large"
                  disabled={isAnalyzingFields || isGeneratingImage}
                  onClick={handleReset}
                >
                  Reset
                </Button>
              </Flex>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
              style={{ display: 'none' }}
            />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', padding: '0 16px' }}>
              <ImageUploadCard
                uploadedImage={uploadedImage}
                isUploading={isUploading}
                imageMetadata={imageMetadata}
                onFileSelect={() => fileInputRef.current?.click()}
                onFileChange={() => fileInputRef.current?.click()}
                onFileRemove={handleReset}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              />

              <FieldsCard
                fields={fields}
                augmentedData={augmentedData}
                isAnalyzing={isAnalyzingFields}
                isGenerating={isGeneratingImage}
                onFieldChange={(field, value) => setFields(prev => ({ ...prev, [field]: value }))}
              />
            </div>

            <GeneratedVariationsSection
              generatedImages={generatedImages}
              generated3DModel={generated3DModel}
              model3DError={model3DError}
              isGenerating={isGeneratingImage}
            />
          </Stack>
        </div>
      </main>
    </div>
  );
}

export default Home;
