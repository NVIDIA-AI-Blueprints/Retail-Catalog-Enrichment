'use client';

import { 
  AppBar, 
  Text, 
  Card, 
  Stack, 
  Grid,
  GridItem,
  Button,
  Flex,
  FormField,
  TextInput,
  TextArea,
  Spinner
} from '@/kui-foundations-react-external';
import Image from 'next/image';
import { useState, useRef } from 'react';

function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [imageMetadata, setImageMetadata] = useState<{
    name: string;
    size: string;
    dimensions?: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  const handleFileUpload = async (file: File) => {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a PNG, JPG, or JPEG file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setIsUploading(true);

    // Simulate upload delay
    setTimeout(() => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new window.Image();
        img.onload = () => {
          setImageMetadata({
            name: file.name,
            size: formatFileSize(file.size),
            dimensions: `${img.width} × ${img.height}`
          });
        };
        img.src = e.target?.result as string;
        setUploadedImage(e.target?.result as string);
        setIsUploading(false);
      };
      reader.readAsDataURL(file);
    }, 1500);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveImage = () => {
    setUploadedImage(null);
    setImageMetadata(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-surface-base">
      {/* App Bar */}
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

      {/* Main Content */}
      <main className="pt-16 bg-surface-base min-h-screen">
        <div className="max-w-7xl mx-auto px-8" style={{ paddingTop: '48px', paddingBottom: '48px' }}>
          <Stack gap="8">
            {/* Main Content Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', padding: '0 16px', marginTop: '24px' }}>
              {/* Left Side - Photo Upload */}
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
                  
                  {/* Hidden File Input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  
                  {/* Image Upload Area */}
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
                          <Button 
                            kind="secondary" 
                            size="small"
                            onClick={handleUploadClick}
                          >
                            Change
                          </Button>
                          <Button 
                            kind="secondary" 
                            size="small"
                            onClick={handleRemoveImage}
                          >
                            Remove
                          </Button>
                        </div>
                      </div>
                      
                      {/* Image Metadata */}
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
                      onClick={handleUploadClick}
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

              {/* Right Side - Fields */}
              <Card>
                <Stack gap="6">
                  <Text kind="title/md" className="text-primary">Fields</Text>
                  
                  <Stack gap="4">
                    {/* Title Field */}
                    <FormField
                      slotLabel="Title"
                    >
                      {(args) => (
                        <TextInput 
                          {...args}
                          placeholder="Enter product title"
                          size="medium"
                        />
                      )}
                    </FormField>

                    {/* Description Field */}
                    <FormField
                      slotLabel="Description"
                    >
                      {(args) => (
                        <TextArea 
                          {...args}
                          placeholder="Enter product description"
                          size="medium"
                          resizeable="manual"
                          attributes={{
                            TextAreaElement: {
                              rows: 3
                            }
                          }}
                        />
                      )}
                    </FormField>

                    {/* Color Field */}
                    <FormField
                      slotLabel="Color"
                    >
                      {(args) => (
                        <TextInput 
                          {...args}
                          placeholder="e.g., Black, Red, Blue"
                          size="medium"
                        />
                      )}
                    </FormField>

                    {/* Tags Field */}
                    <FormField
                      slotLabel="Tags"
                    >
                      {(args) => (
                        <TextInput 
                          {...args}
                          placeholder="e.g., fashion, accessories, luxury"
                          size="medium"
                        />
                      )}
                    </FormField>

                    {/* Price Field */}
                    <FormField
                      slotLabel="Price"
                    >
                      {(args) => (
                        <TextInput 
                          {...args}
                          type="number"
                          placeholder="0.00"
                          size="medium"
                        />
                      )}
                    </FormField>
                  </Stack>
                </Stack>
              </Card>
            </div>

            {/* Generate Button */}
            <div style={{ padding: '0 16px' }}>
              <Stack gap="4" align="center">
                <Button 
                  kind="primary" 
                  size="large" 
                  className="min-w-[200px] nvidia-green-button"
                  disabled={!uploadedImage}
                >
                  {uploadedImage ? 'Generate Enriched Data' : 'Upload Image to Start'}
                </Button>
              </Stack>
            </div>
          </Stack>
        </div>
      </main>
    </div>
  );
}

export default Home;






