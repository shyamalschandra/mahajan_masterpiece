# GitHub Pages Setup Guide

This guide explains how to set up and deploy this project to GitHub Pages using Jekyll.

## Prerequisites

1. A GitHub account
2. Git installed on your local machine
3. Ruby and Bundler installed (for local Jekyll testing)

## Setup Steps

### 1. Create GitHub Repository

**Important**: For the URL `mahajan_masterpiece.github.io` to work, the repository MUST be named exactly `mahajan_masterpiece.github.io`.

1. Go to [GitHub](https://github.com) and create a new repository
2. Repository name: `mahajan_masterpiece.github.io` (for custom domain) OR `mahajan_masterpiece` (for subdirectory URL)
3. Make it public (required for free GitHub Pages)
4. Initialize with README (optional)

**Note**: 
- Repository name `mahajan_masterpiece.github.io` → URL: `https://mahajan_masterpiece.github.io/`
- Repository name `mahajan_masterpiece` → URL: `https://USERNAME.github.io/mahajan_masterpiece/`

### 2. Push Code to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: ECG classification comparison project"

# Add remote (replace USERNAME with your GitHub username and REPO_NAME with repository name)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# If using custom domain format:
# git remote add origin https://github.com/mahajan_masterpiece/mahajan_masterpiece.github.io.git

# Push to main branch
git branch -M main
git push -u origin main
```

### 3. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings**
3. Scroll down to **Pages** section
4. Under **Source**, select:
   - **Branch**: `main` (or `master`)
   - **Folder**: `/ (root)` or `/docs` if using docs folder
5. Click **Save**

### 4. Configure GitHub Pages Settings

GitHub Pages will automatically:
- Detect Jekyll configuration from `_config.yml`
- Build the site using Jekyll
- Deploy based on repository name:
  - `mahajan_masterpiece.github.io` → `https://mahajan_masterpiece.github.io/`
  - `mahajan_masterpiece` → `https://USERNAME.github.io/mahajan_masterpiece/`

### 5. Custom Domain (Optional)

If you want to use a custom domain (not `*.github.io`):

1. In repository Settings → Pages
2. Under **Custom domain**, enter your domain (e.g., `yourdomain.com`)
3. GitHub will automatically create a CNAME file
4. Configure DNS settings to point to GitHub Pages

**Note**: 
- For `username.github.io` format, the repository must be named exactly `username.github.io`
- The repository name determines the base URL automatically

## Local Jekyll Testing

### Install Dependencies

```bash
# Install Ruby (if not installed)
# macOS: brew install ruby
# Ubuntu: sudo apt-get install ruby-full

# Install Bundler
gem install bundler

# Install Jekyll and dependencies
bundle install
```

### Run Local Server

```bash
# Build and serve locally
bundle exec jekyll serve

# Or with live reload
bundle exec jekyll serve --livereload
```

Visit `http://localhost:4000` to view your site locally.

### Build for Production

```bash
# Build static site
bundle exec jekyll build

# Output will be in _site/ directory
```

## File Structure for Jekyll

```
Mahajan_Masterpiece/
├── _config.yml          # Jekyll configuration
├── _layouts/
│   └── default.html     # Main layout template
├── index.md             # Homepage (Markdown)
├── Gemfile              # Ruby dependencies
├── .github/
│   └── workflows/
│       └── jekyll.yml   # GitHub Actions workflow
├── .nojekyll            # Disable Jekyll processing for certain files
└── [other project files]
```

## Important Files

### `_config.yml`
- Contains site configuration
- Defines title, author, description
- Configures Jekyll plugins
- Sets up navigation

### `index.md`
- Main homepage content
- Uses Markdown format
- Rendered using `default.html` layout

### `_layouts/default.html`
- Main HTML template
- Includes header, footer, and styling
- Uses Liquid templating for dynamic content

### `.nojekyll`
- Empty file that tells GitHub Pages to skip Jekyll processing
- Useful if you want to serve static HTML directly

## GitHub Actions Workflow

The `.github/workflows/jekyll.yml` file sets up automatic building:
- Builds site on push to main/master
- Runs on pull requests
- Tests Jekyll build process

## Troubleshooting

### Site Not Updating

1. Check GitHub Actions tab for build errors
2. Verify `_config.yml` syntax is correct
3. Check that `index.md` or `index.html` exists
4. Wait a few minutes for GitHub Pages to rebuild

### Jekyll Build Errors

1. Check Ruby version (should be 3.1+)
2. Run `bundle update` to update dependencies
3. Check `_config.yml` for syntax errors
4. Verify all required files are present

### Custom Domain Issues

1. DNS settings must point to GitHub Pages
2. Repository must be public (for free tier)
3. Wait up to 24 hours for DNS propagation

## Updating the Site

1. Make changes to files
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update website content"
   git push origin main
   ```
3. GitHub Pages will automatically rebuild
4. Changes appear within 1-5 minutes

## Alternative: Static HTML

If you prefer to use the static `index.html` directly:

1. Rename `index.html` to keep it
2. Add `.nojekyll` file (already created)
3. GitHub Pages will serve it as static HTML
4. No Jekyll processing needed

## Repository Settings

For best results:
- Repository visibility: **Public** (required for free GitHub Pages)
- Branch protection: Optional, but recommended for main branch
- GitHub Pages source: **Deploy from a branch** → `main` → `/ (root)`

## URL Structure

After setup, your site will be available at:
- `https://USERNAME.github.io/mahajan_masterpiece/`
- Or custom domain if configured

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Liquid Templating](https://shopify.github.io/liquid/)

## Notes

- GitHub Pages uses Jekyll 4.x by default
- Some Jekyll plugins may not be supported on GitHub Pages
- File size limits: 1GB repository, 100MB per file
- Build time limit: 10 minutes per build
