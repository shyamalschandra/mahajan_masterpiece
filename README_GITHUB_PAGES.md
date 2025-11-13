# Quick Start: GitHub Pages Deployment

## Repository Setup for Project Page

To deploy under `shyamalschandra.github.io/mahajan_masterpiece/`:

### Steps

1. **Create Repository**:
   - Repository name: `mahajan_masterpiece`
   - Owner: `shyamalschandra`
   - Make it public

2. **Push Code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Jekyll site setup"
   git remote add origin https://github.com/shyamalschandra/mahajan_masterpiece.git
   git branch -M main
   git push -u origin main
   ```

3. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Source: Deploy from branch `main` → `/ (root)`
   - Site will be live at: `https://shyamalschandra.github.io/mahajan_masterpiece/`

## Multiple Sites Under Same Account

You can have multiple GitHub Pages sites:

- **Main site**: `shyamalschandra.github.io` (repository: `shyamalschandra.github.io`)
- **Project 1**: `shyamalschandra.github.io/mahajan_masterpiece/` (repository: `mahajan_masterpiece`)
- **Project 2**: `shyamalschandra.github.io/another-project/` (repository: `another-project`)
- **Project 3**: `shyamalschandra.github.io/yet-another/` (repository: `yet-another`)

All can coexist under the same GitHub account!

## Files Structure

- `_config.yml` - Jekyll configuration (configured for project page)
- `_layouts/default.html` - Main layout template
- `index.md` - Homepage (Markdown, processed by Jekyll)
- `index.html` - Static HTML version (backup)
- `Gemfile` - Ruby dependencies
- `.github/workflows/jekyll.yml` - CI/CD workflow

## Local Testing

```bash
# Install dependencies
bundle install

# Run local server (with baseurl)
bundle exec jekyll serve --baseurl /mahajan_masterpiece

# Or use the default (will use baseurl from _config.yml)
bundle exec jekyll serve

# Visit http://localhost:4000/mahajan_masterpiece/
```

## Important Notes

1. **Repository Name**: Can be any name (e.g., `mahajan_masterpiece`)
2. **URL Format**: `https://shyamalschandra.github.io/mahajan_masterpiece/`
3. **Base URL**: Configured in `_config.yml` as `/mahajan_masterpiece`
4. **Public Repository**: Required for free GitHub Pages hosting
5. **Build Time**: First deployment takes 1-5 minutes
6. **Updates**: Push to `main` branch triggers automatic rebuild

## Alternative: Custom Domain

If you want a custom domain (e.g., `mahajan-masterpiece.com`):

1. Create repository: `mahajan_masterpiece`
2. Enable Pages: Settings → Pages → Custom domain: `mahajan-masterpiece.com`
3. Configure DNS settings
4. Update `_config.yml`:
   ```yaml
   url: "https://mahajan-masterpiece.com"
   baseurl: ""
   ```

## Troubleshooting

- **Site not updating**: Check GitHub Actions tab for build errors
- **404 errors**: Verify `index.md` or `index.html` exists
- **Build failures**: Check `_config.yml` syntax
- **Links broken**: Ensure `baseurl` is correctly set in `_config.yml`

For detailed setup instructions, see `GITHUB_PAGES_SETUP.md`.
For information about multiple sites, see `GITHUB_PAGES_MULTIPLE_SITES.md`.
