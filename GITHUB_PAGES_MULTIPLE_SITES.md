# Multiple GitHub Pages Sites Guide

## Understanding GitHub Pages URL Structure

You can have multiple GitHub Pages sites under the same GitHub account (`shyamalschandra`). Here are your options:

### Option 1: Project Pages (Recommended for Multiple Sites)

**URL Format**: `https://shyamalschandra.github.io/mahajan_masterpiece/`

- Repository name: `mahajan_masterpiece` (any name you want)
- Base URL: `https://shyamalschandra.github.io/mahajan_masterpiece/`
- You can have unlimited project pages
- Each project gets its own subdirectory

**Setup**:
1. Create repository: `mahajan_masterpiece`
2. Push code to `main` branch
3. Enable Pages: Settings → Pages → Source: `main` → `/ (root)`
4. Site URL: `https://shyamalschandra.github.io/mahajan_masterpiece/`

### Option 2: User/Organization Site (Only ONE per account)

**URL Format**: `https://shyamalschandra.github.io/`

- Repository name: **MUST be** `shyamalschandra.github.io`
- Base URL: `https://shyamalschandra.github.io/` (root domain)
- Only ONE repository can use this format per GitHub account
- This is your "main" personal/organization site

**Setup**:
1. Create repository: `shyamalschandra.github.io`
2. Push code to `main` branch
3. Enable Pages: Settings → Pages → Source: `main` → `/ (root)`
4. Site URL: `https://shyamalschandra.github.io/`

### Option 3: Custom Domain (Multiple Sites Possible)

**URL Format**: `https://mahajan-masterpiece.com/` (or any custom domain)

- Repository name: Any name (e.g., `mahajan_masterpiece`)
- Requires custom domain and DNS configuration
- Can have multiple custom domains
- Each repository can have its own custom domain

**Setup**:
1. Create repository: `mahajan_masterpiece`
2. Push code to `main` branch
3. Enable Pages: Settings → Pages → Custom domain: `mahajan-masterpiece.com`
4. Configure DNS to point to GitHub Pages
5. Site URL: `https://mahajan-masterpiece.com/`

## Important Notes

### For `mahajan_masterpiece.github.io` URL

**This is NOT possible** under the `shyamalschandra` account because:
- `mahajan_masterpiece.github.io` would require a separate GitHub account or organization named `mahajan_masterpiece`
- GitHub usernames/organization names determine the `*.github.io` domain

**What you CAN do**:
- Use `shyamalschandra.github.io/mahajan_masterpiece/` (project page)
- Use a custom domain like `mahajan-masterpiece.com`
- Create a separate GitHub organization named `mahajan_masterpiece` (if available)

## Recommended Setup for Multiple Sites

### Scenario: Multiple Projects Under `shyamalschandra`

```
Repository: shyamalschandra.github.io
URL: https://shyamalschandra.github.io/
Purpose: Main personal/portfolio site

Repository: mahajan_masterpiece
URL: https://shyamalschandra.github.io/mahajan_masterpiece/
Purpose: ECG classification project

Repository: project2
URL: https://shyamalschandra.github.io/project2/
Purpose: Another project

Repository: project3
URL: https://shyamalschandra.github.io/project3/
Purpose: Yet another project
```

All of these can coexist under the same GitHub account!

## Configuration for Project Pages

When using project pages (Option 1), update `_config.yml`:

```yaml
baseurl: "/mahajan_masterpiece"  # Repository name
url: "https://shyamalschandra.github.io"
```

This ensures all links work correctly with the subdirectory path.

## Example: Setting Up Multiple Sites

### Site 1: Main Portfolio Site
```bash
# Repository: shyamalschandra.github.io
git clone https://github.com/shyamalschandra/shyamalschandra.github.io.git
cd shyamalschandra.github.io
# Add your portfolio content
git push
# Live at: https://shyamalschandra.github.io/
```

### Site 2: ECG Project
```bash
# Repository: mahajan_masterpiece
git clone https://github.com/shyamalschandra/mahajan_masterpiece.git
cd mahajan_masterpiece
# This project's content
git push
# Live at: https://shyamalschandra.github.io/mahajan_masterpiece/
```

### Site 3: Another Project
```bash
# Repository: another-project
git clone https://github.com/shyamalschandra/another-project.git
cd another-project
# Another project's content
git push
# Live at: https://shyamalschandra.github.io/another-project/
```

## Updating Jekyll Configuration for Project Pages

For the `mahajan_masterpiece` repository, your `_config.yml` should have:

```yaml
baseurl: "/mahajan_masterpiece"
url: "https://shyamalschandra.github.io"
```

Then in your templates, use:
- `{{ site.baseurl }}` for internal links
- `{{ '/path' | relative_url }}` for relative URLs

## Summary

✅ **YES**, you can have multiple GitHub Pages sites under `shyamalschandra`:
- One user site: `shyamalschandra.github.io/`
- Unlimited project pages: `shyamalschandra.github.io/project-name/`

❌ **NO**, you cannot have `mahajan_masterpiece.github.io` under `shyamalschandra` account:
- That would require a separate account/organization named `mahajan_masterpiece`
- Use `shyamalschandra.github.io/mahajan_masterpiece/` instead

✅ **YES**, you can use custom domains for any repository

