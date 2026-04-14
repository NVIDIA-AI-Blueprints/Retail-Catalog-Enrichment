'use client';

import { Stack } from '@/kui-foundations-react-external';
import { useEffect, useState, useRef } from 'react';
import { Nebula } from '@/kui-foundations-react-external/nebula';
import { Header } from '@/components/Header';
import { ImageUploadCard } from '@/components/ImageUploadCard';
import { FieldsCard } from '@/components/FieldsCard';
import { AdvancedOptionsCard } from '@/components/AdvancedOptionsCard';
import { GeneratedVariationsSection } from '@/components/GeneratedVariationsSection';
import { ProductFields, AugmentedData, PolicyDocument, PolicyUploadResult, SUPPORTED_LOCALES } from '@/types';
import { analyzeImage, clearPolicies, generateImageVariation, generate3DModel, listPolicies, prepareProductData, uploadPolicies } from '@/lib/api';


function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzingFields, setIsAnalyzingFields] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [locale, setLocale] = useState<string>('en-US');
  const [loadedPolicies, setLoadedPolicies] = useState<PolicyDocument[]>([]);
  const [policyUploadResults, setPolicyUploadResults] = useState<PolicyUploadResult[]>([]);
  const [policyUploadError, setPolicyUploadError] = useState<string | null>(null);
  const [isUploadingPolicies, setIsUploadingPolicies] = useState(false);
  const [fields, setFields] = useState<ProductFields>({
    title: '',
    description: '',
    color: '',
    categories: '',
    tags: '',
    price: '',
    brandInstructions: ''
  });
  const [augmentedData, setAugmentedData] = useState<AugmentedData | null>(null);
  const [generatedImages, setGeneratedImages] = useState<(string | null)[]>([null, null]);
  const [qualityScores, setQualityScores] = useState<(number | null)[]>([null, null]);
  const [qualityIssues, setQualityIssues] = useState<(string[] | null)[]>([null, null]);
  const [generated3DModel, setGenerated3DModel] = useState<string | null>(null);
  const [model3DError, setModel3DError] = useState<string | null>(null);
  const [enableVariation1, setEnableVariation1] = useState<boolean>(true);
  const [enableVariation2, setEnableVariation2] = useState<boolean>(true);
  const [enable3D, setEnable3D] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const policyFileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void refreshPolicies({ silent: true });
  }, []);

  const handleFileUpload = async (file: File) => {
    if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
      alert('Please upload a PNG, JPG, or JPEG file');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    // Clear image-dependent results when replacing, but preserve settings
    setAugmentedData(null);
    setGeneratedImages([null, null]);
    setQualityScores([null, null]);
    setQualityIssues([null, null]);
    setGenerated3DModel(null);
    setModel3DError(null);

    setIsUploading(true);
    setUploadedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new window.Image();
      img.onload = () => {
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
    setAugmentedData(null);
    setGeneratedImages([null, null]);
    setQualityScores([null, null]);
    setQualityIssues([null, null]);
    setGenerated3DModel(null);
    setModel3DError(null);
    setLocale('en-US');
    setPolicyUploadResults([]);
    setPolicyUploadError(null);
    setFields({ title: '', description: '', color: '', categories: '', tags: '', price: '', brandInstructions: '' });
    setEnableVariation1(true);
    setEnableVariation2(true);
    setEnable3D(true);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (policyFileInputRef.current) policyFileInputRef.current.value = '';
  };

  const refreshPolicies = async ({ silent = false }: { silent?: boolean } = {}) => {
    try {
      const documents = await listPolicies();
      setLoadedPolicies(documents);
      if (!silent) {
        setPolicyUploadError(null);
      }
    } catch (error) {
      console.error('Error loading policy library:', error);
      setLoadedPolicies([]);
      if (!silent) {
        setPolicyUploadError(error instanceof Error ? error.message : 'Failed to load policy library');
      }
    }
  };

  const handlePolicyFilesUpload = async (files: FileList | null) => {
    const selectedFiles = Array.from(files || []);
    if (selectedFiles.length === 0) {
      return;
    }

    const invalidFile = selectedFiles.find((file) => file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf'));
    if (invalidFile) {
      setPolicyUploadError(`"${invalidFile.name}" is not a PDF.`);
      return;
    }

    try {
      setIsUploadingPolicies(true);
      setPolicyUploadError(null);
      const response = await uploadPolicies(selectedFiles, locale);
      setLoadedPolicies(response.documents || []);
      setPolicyUploadResults(response.results || []);
    } catch (error) {
      console.error('Error uploading policy PDFs:', error);
      setPolicyUploadError(error instanceof Error ? error.message : 'Failed to upload policy PDFs');
    } finally {
      setIsUploadingPolicies(false);
      if (policyFileInputRef.current) {
        policyFileInputRef.current.value = '';
      }
    }
  };

  const handleClearPolicyLibrary = async () => {
    if (!window.confirm('This will clear all loaded policy PDFs and embeddings. Continue?')) {
      return;
    }

    try {
      setIsUploadingPolicies(true);
      setPolicyUploadError(null);
      await clearPolicies();
      setLoadedPolicies([]);
      setPolicyUploadResults([]);
      setAugmentedData(prev => prev ? { ...prev, policyDecision: undefined } : prev);
    } catch (error) {
      console.error('Error clearing policy library:', error);
      setPolicyUploadError(error instanceof Error ? error.message : 'Failed to clear policy library');
    } finally {
      setIsUploadingPolicies(false);
    }
  };

  const handleGenerate = async () => {
    if (!uploadedFile) return;

    setAugmentedData(null);
    setGeneratedImages([null, null]);
    setQualityScores([null, null]);
    setQualityIssues([null, null]);
    setGenerated3DModel(null);
    setModel3DError(null);
    
    try {
      setIsAnalyzingFields(true);
      const productData = prepareProductData(fields);
      
      const analyzeData = await analyzeImage({ 
        file: uploadedFile, 
        locale, 
        productData,
        brandInstructions: fields.brandInstructions
      });
      
      setAugmentedData({
        title: analyzeData.title || '',
        description: analyzeData.description || '',
        colors: analyzeData.colors || [],
        tags: analyzeData.tags || [],
        categories: analyzeData.categories || [],
        policyDecision: analyzeData.policyDecision
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

      const tasks = [];
      
      if (enableVariation1) {
        tasks.push(
          (async () => {
            try {
              const result = await generateImageVariation(variationParams);
              setGeneratedImages(prev => [result.imageUrl, prev[1]]);
              setQualityScores(prev => [result.qualityScore, prev[1]]);
              setQualityIssues(prev => [result.qualityIssues, prev[1]]);
            } catch (error) {
              console.error('Error generating variation 1:', error);
            }
          })()
        );
      }
      
      if (enableVariation2) {
        tasks.push(
          (async () => {
            try {
              const result = await generateImageVariation(variationParams);
              setGeneratedImages(prev => [prev[0], result.imageUrl]);
              setQualityScores(prev => [prev[0], result.qualityScore]);
              setQualityIssues(prev => [prev[0], result.qualityIssues]);
            } catch (error) {
              console.error('Error generating variation 2:', error);
            }
          })()
        );
      }
      
      if (enable3D) {
        tasks.push(
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
        );
      }
      
      await Promise.all(tasks);
    } catch (error) {
      console.error('Generation error:', error);
      alert(error instanceof Error ? error.message : 'Failed to generate enriched data');
    } finally {
      setIsAnalyzingFields(false);
      setIsGeneratingImage(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface-base relative">
      {/* Nebula Background */}
      <div 
        className="pointer-events-none" 
        style={{ 
          position: 'fixed', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          width: '100vw',
          height: '100vh',
          zIndex: 0,
          overflow: 'hidden'
        }}
      >
        <div style={{ width: '100%', height: '100%' }}>
          <Nebula variant="ambient" />
        </div>
      </div>
      
      {/* Top Green Gradient Overlay */}
      <div 
        className="pointer-events-none"
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: '500px',
          background: 'linear-gradient(80.22deg, #BFF230 1.49%, #7CD7FE 99.95%)',
          opacity: 0.12,
          zIndex: 0,
          maskImage: 'radial-gradient(ellipse 150% 120% at top, black 0%, black 30%, transparent 70%)',
          WebkitMaskImage: 'radial-gradient(ellipse 150% 120% at top, black 0%, black 30%, transparent 70%)'
        }}
      />
      
      {/* Bottom Green Gradient Overlay */}
      <div 
        className="pointer-events-none"
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: '300px',
          background: 'linear-gradient(80.22deg, #BFF230 1.49%, #7CD7FE 99.95%)',
          opacity: 0.12,
          zIndex: 0,
          maskImage: 'radial-gradient(ellipse 120% 130% at bottom, black 0%, black 25%, transparent 60%)',
          WebkitMaskImage: 'radial-gradient(ellipse 120% 130% at bottom, black 0%, black 25%, transparent 60%)'
        }}
      />
      
      {/* Content */}
      <div className="relative" style={{ zIndex: 1 }}>
        <Header />
        
        <main className="pt-16 min-h-screen">
        <div className="max-w-7xl mx-auto px-8" style={{ paddingTop: '48px', paddingBottom: '48px' }}>
          <Stack gap="8">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              style={{ display: 'none' }}
            />
            <input
              ref={policyFileInputRef}
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={(e) => {
                void handlePolicyFilesUpload(e.target.files);
              }}
              style={{ display: 'none' }}
            />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', padding: '0 16px' }}>
              <ImageUploadCard
                uploadedImage={uploadedImage}
                isUploading={isUploading}
                locale={locale}
                localeOptions={SUPPORTED_LOCALES}
                isAnalyzingFields={isAnalyzingFields}
                isGeneratingImage={isGeneratingImage}
                onFileSelect={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onLocaleChange={setLocale}
                onGenerate={handleGenerate}
                onReset={handleReset}
              />

              <FieldsCard
                fields={fields}
                augmentedData={augmentedData}
                isAnalyzing={isAnalyzingFields}
                isGenerating={isGeneratingImage}
                onFieldChange={(field, value) => setFields(prev => ({ ...prev, [field]: value }))}
              />
            </div>

            <div style={{ padding: '0 16px' }}>
              <AdvancedOptionsCard
                brandInstructions={fields.brandInstructions}
                loadedPolicies={loadedPolicies}
                policyUploadResults={policyUploadResults}
                policyUploadError={policyUploadError}
                isUploadingPolicies={isUploadingPolicies}
                enableVariation1={enableVariation1}
                enableVariation2={enableVariation2}
                enable3D={enable3D}
                isAnalyzingFields={isAnalyzingFields}
                isGeneratingImage={isGeneratingImage}
                onBrandInstructionsChange={(value) => setFields(prev => ({ ...prev, brandInstructions: value }))}
                onPolicyFileSelect={() => policyFileInputRef.current?.click()}
                onClearPolicyLibrary={() => {
                  void handleClearPolicyLibrary();
                }}
                onEnableVariation1Change={setEnableVariation1}
                onEnableVariation2Change={setEnableVariation2}
                onEnable3DChange={setEnable3D}
              />
            </div>

            <GeneratedVariationsSection
              generatedImages={generatedImages}
              qualityScores={qualityScores}
              qualityIssues={qualityIssues}
              generated3DModel={generated3DModel}
              model3DError={model3DError}
              isGenerating={isGeneratingImage}
            />
          </Stack>
        </div>
      </main>
      </div>
    </div>
  );
}

export default Home;
