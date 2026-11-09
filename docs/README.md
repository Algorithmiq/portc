# portc documentation site

A [Fumadocs](https://fumadocs.dev) (Next.js) application, exported as static
files and deployed to GitHub Pages by `.github/workflows/docpages.yml`.

- `content/docs/` holds the hand-written MDX pages; `meta.json` files order the
  sidebar.
- `notebooks/<name>/<name>.ipynb` are the tutorials, executed at build time.
- `scripts/` generate the API reference from the docstrings and the tutorial
  pages from the notebooks.
- `src/` is the Next.js app: layout, components and routes.

Build it from the repository root with Node.js 22 or later:

```bash
uv sync --group docs
cd docs
npm ci
uv run python scripts/gen_api_dump.py portc -d .
node scripts/generate-api.mjs
uv run python scripts/notebooks_to_mdx.py
npm run dev     # live preview on http://localhost:3000
npm run build   # static export in out/
```

The Documenting page of the site (`content/docs/contributing/documenting.mdx`)
covers writing pages, cross-references, citations and tutorials.
