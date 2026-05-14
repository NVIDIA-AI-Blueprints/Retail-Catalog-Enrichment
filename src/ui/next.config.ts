import type { NextConfig } from 'next';
import path from 'node:path';

const nextConfig: NextConfig = {
  output: 'standalone',
  outputFileTracingRoot: path.resolve(__dirname),
  transpilePackages: ['@kui/foundations-react-external'],
  experimental: {
    externalDir: true,
  },
  webpack: (config) => {
    config.resolve.extensionAlias = {
      '.js': ['.js', '.ts', '.tsx', '.jsx'],
    };
    return config;
  },
};

export default nextConfig;




