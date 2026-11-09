// The static export is served from a subpath on GitHub Pages
// (algorithmiq.github.io/portc), which Next.js applies to its own links and
// assets but not to URLs we fetch by hand; prefix those with `basePath`.
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// Absolute URL of the deployed site, for the Open Graph images.
export const siteUrl = `https://algorithmiq.github.io${basePath}`;

export const appName = 'portc';
export const docsRoute = '/';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'Algorithmiq',
  repo: 'portc',
  branch: 'main',
};
