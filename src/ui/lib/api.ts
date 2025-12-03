import { ProductFields, AugmentedData, NIMHealthStatus } from '../types';

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

  return response.json();
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
  
  if (data.quality_issues && data.quality_issues.length > 0) {
    console.log(`[Quality Check] Score: ${data.quality_score}% - Issues found:`, data.quality_issues);
  }
  
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

  console.log('[3D API] Starting 3D model generation request...');
  const startTime = Date.now();

  const response = await fetch(`${API_BASE}/generate/3d`, {
    method: 'POST',
    body: formData,
    // Increase timeout for large responses (2 minutes)
    signal: AbortSignal.timeout(120000)
  });

  const elapsed = Date.now() - startTime;
  console.log(`[3D API] Response received in ${elapsed}ms, status: ${response.status}`);

  if (!response.ok) {
    const error = await response.json();
    console.error('[3D API] Error response:', error);
    throw new Error(error.detail || 'Failed to generate 3D model');
  }

  const contentLength = response.headers.get('content-length');
  const contentType = response.headers.get('content-type');
  console.log(`[3D API] Response headers - Content-Type: ${contentType}, Content-Length: ${contentLength}`);

  let data;
  try {
    data = await response.json();
    console.log(`[3D API] JSON parsed successfully, has glb_base64: ${!!data.glb_base64}, artifact_id: ${data.artifact_id}`);
    
    if (data.glb_base64) {
      const base64Length = data.glb_base64.length;
      const estimatedSizeMB = (base64Length * 0.75) / (1024 * 1024);
      console.log(`[3D API] GLB base64 length: ${base64Length} chars (~${estimatedSizeMB.toFixed(2)} MB decoded)`);
    } else {
      console.error('[3D API] Response missing glb_base64 field. Full response:', data);
    }
  } catch (parseError) {
    console.error('[3D API] Failed to parse JSON response:', parseError);
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

