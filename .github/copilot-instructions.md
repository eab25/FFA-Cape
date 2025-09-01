# FFA Trading Streamlit Application

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

FFA-Cape is a Streamlit web application for FFA Trading data visualization. It allows users to upload CSV files containing trading data and view summary statistics and line plots of numeric columns.

## Working Effectively

### Initial Setup and Dependencies
Run these commands in sequence to set up the development environment:

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install matplotlib (missing from requirements.txt but required by the app)
pip3 install matplotlib
```

**TIMING**: Dependencies install takes 15-20 seconds total. NEVER CANCEL during installation.

### Building and Running
There is no build step required for this Streamlit application. To run the application:

```bash
# Start the Streamlit application
streamlit run streamlit_app.py
```

**TIMING**: Streamlit starts in under 5 seconds. Application will be available at http://localhost:8501

**Alternative run command for headless environments:**
```bash
streamlit run streamlit_app.py --server.headless true --server.port 8501
```

### Testing the Application
**CRITICAL**: Always manually validate the application functionality after making changes by running through this complete scenario:

1. **Start the application**: `streamlit run streamlit_app.py`
2. **Verify startup**: Application should start in under 5 seconds and be available at http://localhost:8501
3. **Test file upload**: Upload the sample CSV file "Baltic TC index all vsl" (included in repository)
4. **Verify the following functionality**:
   - File uploads successfully without errors
   - Data preview displays correctly (should show 1414 rows, 5 columns)
   - Summary statistics appear with min/max/mean values
   - Line plot selector shows numeric columns (C5TC, HS7TC, P4TC, S10TC)
   - Line plot renders correctly when a column is selected
   - No error messages in the Streamlit interface

**Sample Data**: Use the included "Baltic TC index all vsl" file for testing - it contains 1414 rows of Baltic trading data with Date, C5TC, HS7TC, P4TC, and S10TC columns.

**Quick Validation Commands**:
```bash
# Verify app responds
curl -s http://localhost:8501 | grep -q "Streamlit" && echo "✓ App is running" || echo "✗ App not responding"

# Verify sample data is available
head -3 "Baltic TC index all vsl"
```

## Validation Requirements

### Before Making Changes
Always validate the current state works:
```bash
# Test all imports work correctly
python3 -c "import streamlit as st; import pandas as pd; import matplotlib.pyplot as plt; print('All imports successful')"

# Test CSV loading works
python3 -c "import pandas as pd; df = pd.read_csv('Baltic TC index all vsl'); print(f'Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns')"

# Validate Python syntax
python3 -m py_compile streamlit_app.py
```

### After Making Changes
ALWAYS run through the complete testing scenario described above. The application must:
- Start without errors
- Accept CSV file uploads
- Display data previews correctly
- Generate visualizations successfully

## Common Issues and Solutions

### Missing matplotlib
If you see `ModuleNotFoundError: No module named 'matplotlib'`:
```bash
pip3 install matplotlib
```

### Port Already in Use
If port 8501 is already in use:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### CSV Upload Issues
The application expects CSV files with numeric columns for plotting. Test with the included "Baltic TC index all vsl" file to verify functionality.

## Repository Structure

### Key Files
- `streamlit_app.py` - Main application file containing upload, analysis, and visualization logic
- `requirements.txt` - Python dependencies (NOTE: matplotlib missing, install separately)
- `Baltic TC index all vsl` - Sample CSV data file for testing (1414 rows, 5 columns)
- `.streamlit/config.toml` - Streamlit theme configuration
- `.devcontainer/devcontainer.json` - VS Code dev container setup (Python 3.11-bullseye)

### File Listing
```
.
├── .devcontainer /
│   └── devcontainer.json
├── .git/
├── .streamlit/
│   └── config.toml
├── Baltic TC index all vsl
├── README.md
├── requirements.txt
└── streamlit_app.py
```

### Requirements.txt Contents
```
streamlit
pandas
numpy
```

**IMPORTANT**: matplotlib is required but missing from requirements.txt - always install it separately.

## Development Environment

### Python Environment
- **Python Version**: Compatible with Python 3.11+ (dev container uses 3.11-bullseye)
- **Package Manager**: pip3
- **Virtual Environment**: Not required but recommended for development

### Dev Container Support
The repository includes VS Code dev container configuration:
- Pre-configured Python 3.11 environment
- Automatic dependency installation via postCreateCommand
- Port forwarding for Streamlit (8501)
- Auto-start application via postAttachCommand

## No Testing Framework
This repository does not include automated tests. Always perform manual validation using the testing scenario described above.

## No Linting or Formatting Tools
This repository does not include linting tools like flake8 or formatting tools like black. Code style is maintained manually. You can use basic Python syntax checking with:
```bash
python3 -m py_compile streamlit_app.py
```

## No CI/CD
There are no GitHub workflows or CI/CD pipelines configured for this repository.

## Application Architecture

### Core Functionality
1. **File Upload**: `load_csv()` function handles CSV file reading with error handling
2. **Data Analysis**: `show_summary()` displays pandas DataFrame.describe() statistics  
3. **Visualization**: `plot_column()` creates matplotlib line plots of selected numeric columns
4. **Main Interface**: `main()` orchestrates the Streamlit UI and workflow

### Data Flow
1. User uploads CSV file via Streamlit file_uploader
2. CSV is loaded into pandas DataFrame
3. Data preview and summary statistics are displayed
4. User selects numeric column for visualization
5. Matplotlib generates line plot displayed via st.pyplot()

### Key Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization (install separately)
- **numpy**: Numerical computing (dependency of pandas)