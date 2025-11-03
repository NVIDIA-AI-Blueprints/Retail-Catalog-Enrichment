import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
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





