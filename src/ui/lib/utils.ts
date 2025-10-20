export const formatFileSize = (bytes: number): string => 
  bytes < 1024 ? `${bytes} bytes` : 
  bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : 
  `${(bytes / 1048576).toFixed(1)} MB`;

