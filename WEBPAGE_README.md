# Project Webpage

This directory contains the project webpage (`index.html`) that provides a comprehensive overview of the ECG classification comparison project.

## Viewing the Webpage

### Option 1: Open Directly in Browser
Simply open `index.html` in your web browser:
```bash
# On macOS
open index.html

# On Linux
xdg-open index.html

# On Windows
start index.html
```

### Option 2: Using Python HTTP Server
For a better experience (especially for testing links), use a local web server:

```bash
# Python 3
python3 -m http.server 8000

# Then open in browser:
# http://localhost:8000/index.html
```

### Option 3: Using Node.js HTTP Server
```bash
# Install http-server globally (if not installed)
npm install -g http-server

# Run server
http-server

# Then open in browser:
# http://localhost:8080/index.html
```

## Webpage Features

The webpage includes:

1. **Header Section**
   - Project title and subtitle
   - Author information (Shyamal Suhana Chandra)
   - Affiliation (Sapana Micro Software, Research Division)
   - Quick links to paper, presentation, README, benchmark guide, project summary, and code repository

2. **Abstract Section**
   - Complete project abstract describing all seven models

3. **Seven Deep Learning Architectures**
   - Interactive model cards with key characteristics
   - Hover effects for better user experience

4. **Comprehensive Comparison**
   - SVG visualization of architecture vs. performance
   - Detailed comparison table
   - Trade-offs visualization (Accuracy vs. Efficiency)

5. **Key Features Section**
   - Highlighted features of the project

6. **Key Findings Section**
   - Summary of important results and insights

7. **Quick Start Guide**
   - Installation instructions
   - Code examples for running benchmarks

8. **Citation Section**
   - BibTeX citation format

9. **References Section**
   - All cited papers and sources

10. **Footer**
    - Author and affiliation information
    - Funding source
    - Acknowledgments to Carl Vondrick's Video Policy project
    - Copyright notice

## Customization

To customize the webpage:

1. **Colors**: Edit the CSS color variables in the `<style>` section
   - Primary color: `#667eea` (purple-blue gradient)
   - Secondary color: `#764ba2` (purple)

2. **Content**: Modify the HTML sections to update text, add new sections, or change structure

3. **SVG Charts**: The SVG visualizations can be customized by editing the SVG elements in the comparison section

4. **Links**: Update the links in the header section to point to your actual repository URLs

## Notes

- The webpage is designed to be responsive and works on desktop, tablet, and mobile devices
- All SVG charts are embedded directly in the HTML for easy customization
- The design is inspired by the Video Policy project webpage from Columbia University
- The page includes proper citations and acknowledgments

## Browser Compatibility

The webpage is compatible with:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Future Enhancements

Potential additions:
- Interactive JavaScript charts (using Chart.js or D3.js)
- Embedded video demonstrations
- Interactive model comparison tool
- Live benchmark results
- Model architecture diagrams

