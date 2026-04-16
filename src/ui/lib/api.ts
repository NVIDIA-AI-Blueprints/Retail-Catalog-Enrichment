import { ProductFields, AugmentedData, NIMHealthStatus, PolicyDocument, PolicyUploadResult } from '../types';

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

interface AnalyzeParams {
  file: File;
  locale: string;
  productData?: any;
  brandInstructions?: string;
}

export async function analyzeImage({ file, locale, productData, brandInstructions }: AnalyzeParams): Promise<AugmentedData> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('locale', locale);
  if (productData) {
    formData.append('product_data', JSON.stringify(productData));
  }
  if (brandInstructions) {
    formData.append('brand_instructions', brandInstructions);
  }

  const response = await fetch(`${API_BASE}/vlm/analyze`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to analyze image');
  }

  const data = await response.json();
  return {
    ...data,
    policyDecision: data.policy_decision
  };
}

interface GenerateFaqsParams {
  title: string;
  description: string;
  categories: string[];
  tags: string[];
  colors: string[];
  locale: string;
}

export async function generateFaqs(params: GenerateFaqsParams): Promise<{ question: string; answer: string }[]> {
  const formData = new FormData();
  formData.append('title', params.title);
  formData.append('description', params.description);
  formData.append('categories', JSON.stringify(params.categories));
  formData.append('tags', JSON.stringify(params.tags));
  formData.append('colors', JSON.stringify(params.colors));
  formData.append('locale', params.locale);

  const response = await fetch(`${API_BASE}/vlm/faqs`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate FAQs');
  }

  const data = await response.json();
  return data.faqs || [];
}

export async function listPolicies(): Promise<PolicyDocument[]> {
  const response = await fetch(`${API_BASE}/policies`, { method: 'GET' });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to load policy library');
  }

  const data = await response.json();
  return data.documents || [];
}

export async function uploadPolicies(files: File[], locale: string): Promise<{ documents: PolicyDocument[]; results: PolicyUploadResult[] }> {
  const formData = new FormData();
  formData.append('locale', locale);
  for (const file of files) {
    formData.append('files', file);
  }

  const response = await fetch(`${API_BASE}/policies`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload policy PDFs');
  }

  return response.json();
}

export async function clearPolicies(): Promise<void> {
  const response = await fetch(`${API_BASE}/policies`, { method: 'DELETE' });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to clear policy library');
  }
}

interface GenerateVariationParams {
  file: File;
  locale: string;
  title: string;
  description: string;
  categories: string[];
  tags: string[];
  colors: string[];
  enhancedProduct?: any;
}

export async function generateImageVariation(params: GenerateVariationParams): Promise<{ imageUrl: string | null, qualityScore: number | null, qualityIssues: string[] }> {
  const formData = new FormData();
  formData.append('image', params.file);
  formData.append('locale', params.locale);
  formData.append('title', params.title);
  formData.append('description', params.description);
  formData.append('categories', JSON.stringify(params.categories));
  formData.append('tags', JSON.stringify(params.tags));
  formData.append('colors', JSON.stringify(params.colors));
  if (params.enhancedProduct) {
    formData.append('enhanced_product', JSON.stringify(params.enhancedProduct));
  }

  const response = await fetch(`${API_BASE}/generate/variation`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate variation');
  }

  const data = await response.json();

  return {
    imageUrl: data.generated_image_b64 ? `data:image/png;base64,${data.generated_image_b64}` : null,
    qualityScore: data.quality_score !== undefined && data.quality_score !== null ? data.quality_score : null,
    qualityIssues: data.quality_issues || []
  };
}

export async function generate3DModel(file: File): Promise<string | null> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('return_json', 'true');

  const response = await fetch(`${API_BASE}/generate/3d`, {
    method: 'POST',
    body: formData,
    // Increase timeout for large responses (2 minutes)
    signal: AbortSignal.timeout(120000)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate 3D model');
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('Failed to parse 3D model response');
  }

  return data.glb_base64 ? `data:model/gltf-binary;base64,${data.glb_base64}` : null;
}

export function prepareProductData(fields: ProductFields) {
  const data: any = {};
  
  if (fields.title && fields.title.trim()) {
    data.title = fields.title.trim();
  }
  
  if (fields.description && fields.description.trim()) {
    data.description = fields.description.trim();
  }
  
  if (fields.categories && fields.categories.trim()) {
    const categories = fields.categories.split(',')
      .map(c => c.trim())
      .filter(c => c !== '');
    if (categories.length > 0) {
      data.categories = categories;
    }
  }
  
  if (fields.tags && fields.tags.trim()) {
    const tags = fields.tags.split(',')
      .map(t => t.trim())
      .filter(t => t !== '');
    if (tags.length > 0) {
      data.tags = tags;
    }
  }
  
  if (fields.price && fields.price.trim()) {
    const price = parseFloat(fields.price);
    if (!isNaN(price)) {
      data.price = price;
    }
  }
  
  return Object.keys(data).length > 0 ? data : null;
}

export async function checkNIMHealth(): Promise<NIMHealthStatus> {
  try {
    const response = await fetch(`${API_BASE}/health/nims`, {
      method: 'GET'
    });

    if (!response.ok) {
      throw new Error('Failed to check NIM health');
    }

    return response.json();
  } catch (error) {
    console.error('Error checking NIM health:', error);
    return {
      vlm: 'unhealthy',
      llm: 'unhealthy',
      flux: 'unhealthy',
      trellis: 'unhealthy'
    };
  }
}
