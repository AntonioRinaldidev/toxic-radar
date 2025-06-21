#!/bin/bash
# universal_setup.sh - Universal setup using requirements.txt

set -e

echo "🌍 Universal AI Paraphraser Setup"
echo "🔧 Auto-detecting system and installing from requirements.txt..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Functions for colored logs
log_info() { echo -e "${BLUE}+  $1${NC}"; }
log_success() { echo -e "${GREEN}+  $1${NC}"; }
log_warning() { echo -e "${YELLOW}?  $1${NC}"; }
log_error() { echo -e "${RED}-  $1${NC}"; }
log_feature() { echo -e "${PURPLE}! $1${NC}"; }

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Detect Python command
detect_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        return 1
    fi
}

# Main detection
OS_TYPE=$(detect_os)
PYTHON_CMD=$(detect_python)

log_info "System Detection:"
log_info "  OS: $OS_TYPE"
log_info "  Python: $PYTHON_CMD"

# Validate Python
if [ -z "$PYTHON_CMD" ]; then
    log_error "Python 3 is required but not found"
    log_info "Please install Python 3:"
    case $OS_TYPE in
        "linux") log_info "  sudo apt update && sudo apt install python3 python3-pip" ;;
        "macos") log_info "  brew install python" ;;
        "windows") log_info "  Download from https://python.org" ;;
    esac
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
log_success "Python found: $PYTHON_VERSION"

# Set up virtual environment
setup_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_info "Setting up virtual environment..."
        if [[ ! -d "venv" ]]; then
            $PYTHON_CMD -m venv venv
            log_success "Virtual environment created"
        fi
        
        # Activate virtual environment (OS-specific)
        case $OS_TYPE in
            "windows")
                source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
                ;;
            *)
                source venv/bin/activate
                ;;
        esac
        
        log_success "Virtual environment activated"
    else
        log_info "Virtual environment already active"
    fi
}

# Create requirements.txt if it doesn't exist
create_requirements() {
    if [[ ! -f "requirements.txt" ]]; then
        log_info "Creating universal requirements.txt..."
        
        cat > requirements.txt << 'EOF'
# Universal AI Paraphraser Requirements
# Compatible with all systems (Windows, Linux, macOS)

# NumPy compatibility fix (CRITICAL - prevents crashes)
numpy>=1.21.0,<2.0.0

# PyTorch (CPU version - works on all systems)
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

#CORE AI/ML libraries
protobuf>=3.20.0,<5.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
detoxify>=0.5.0
tokenizers>=0.13.0

# Web framework
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0

# System utilities
psutil>=5.9.0
requests>=2.28.0

# Development/testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
EOF
        
        log_success "Universal requirements.txt created"
    else
        log_info "requirements.txt already exists"
    fi
}

# Install dependencies from requirements.txt
install_dependencies() {
    log_feature "Installing dependencies from requirements.txt..."
    
    # Upgrade pip first (important for compatibility)
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with system-specific optimizations
    log_info "Installing PyTorch optimized for $OS_TYPE..."
    case $OS_TYPE in
        "linux")
            # Check for NVIDIA GPU on Linux
            if command -v nvidia-smi &> /dev/null; then
                log_feature "NVIDIA GPU detected - installing CUDA PyTorch"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            else
                log_info "Installing CPU PyTorch for Linux"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            fi
            ;;
        "windows")
            # Windows might have NVIDIA GPU
            log_info "Installing PyTorch for Windows (will auto-detect GPU)"
            pip install torch torchvision torchaudio
            ;;
        "macos")
            # macOS always uses CPU or MPS
            log_info "Installing CPU PyTorch for macOS"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
        *)
            # Fallback to CPU
            log_info "Installing CPU PyTorch (fallback)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    # Install all other dependencies from requirements.txt
    log_info "Installing remaining dependencies..."
    pip install -r requirements.txt --upgrade
    
    log_success "All dependencies installed successfully!"
}

# Test installation
test_installation() {
    log_info "Testing installation..."
    
    $PYTHON_CMD -c "
import sys
import importlib.metadata
import re

def check_package(name, version_spec=None, warn_only=False):
    try:
        version = importlib.metadata.version(name)
        print(f'✅ {name} {version}')
        
        if version_spec:
            # Simple check for common version specs (>=, <, ==)
            # This is a basic check, a more robust solution would parse PEP 440
            match = re.match(r'([<>=!~]+)(\d.*)', version_spec)
            if match:
                op, required_version = match.groups()
                
                # Convert version strings to tuples of integers for comparison
                current_v_parts = tuple(map(int, (version.split('.'))))
                required_v_parts = tuple(map(int, (required_version.split('.'))))

                is_ok = True
                if op == '>=':
                    is_ok = current_v_parts >= required_v_parts
                elif op == '<':
                    is_ok = current_v_parts < required_v_parts
                elif op == '==':
                    is_ok = current_v_parts == required_v_parts
                # Add more operators if needed
                
                if not is_ok:
                    status = '⚠️  WARNING' if warn_only else '❌ ERROR'
                    print(f'{status}: {name} version {version} does not meet requirement {version_spec}')
                    if not warn_only:
                        sys.exit(1)
            else:
                print(f'⚠️  WARNING: Could not parse version specification for {name}: {version_spec}')

    except importlib.metadata.PackageNotFoundError:
        print(f'❌ {name} not found')
        sys.exit(1)
    except Exception as e:
        print(f'❌ Failed to check {name}: {e}')
        sys.exit(1)

print(f'✅ Python {sys.version}')
print('--- Checking Core Dependencies ---')

# NumPy
check_package('numpy', '>=1.21.0')
# Special check for NumPy 2.x
try:
    import numpy as np
    if np.__version__.startswith('2.'):
        print('⚠️  WARNING: NumPy 2.x detected. While >=1.21.0 is specified, 2.x might have breaking changes for older ML libraries.')
        print('🎯 Consider pinning to <2.0.0 if you encounter issues.')
    else:
        print('🎯 NumPy 1.x - perfect for most ML libraries')
except ImportError:
    pass # Already handled by check_package

# PyTorch
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    if torch.__version__.startswith('2.'):
        print('🎯 PyTorch 2.x - perfect for this setup')
    else:
        print('⚠️  WARNING: PyTorch version is not 2.x, but meets >=2.0.0. Ensure full compatibility.')
    
    if torch.cuda.is_available():
        print(f'🚀 CUDA available: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('🍎 Metal Performance Shaders (MPS) available')
    else:
        print('💻 Using CPU (optimized for this system)')
        
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    sys.exit(1)

check_package('torchvision', '>=0.15.0')
check_package('torchaudio', '>=2.0.0')

print('--- Checking AI/ML Libraries ---')
check_package('protobuf', '>=3.20.0') 
check_package('transformers', '>=4.30.0')
check_package('sentence-transformers', '>=2.2.0')
check_package('detoxify', '>=0.5.0')
check_package('tokenizers', '>=0.13.0')

print('--- Checking Web Framework ---')
check_package('fastapi', '>=0.100.0')
# uvicorn[standard] is tricky to check directly for the 'standard' part
check_package('uvicorn', '>=0.20.0') 
check_package('pydantic', '>=2.0.0')

print('--- Checking System Utilities ---')
check_package('psutil', '>=5.9.0')
check_package('requests', '>=2.28.0')

print('--- Checking Development/Testing Libraries ---')
check_package('pytest', '>=7.0.0')
check_package('pytest-asyncio', '>=0.21.0')


print('')
print('🎉 Universal installation check complete!')
print('✅ All specified packages found and meet minimum version requirements.')
print('🔒 NumPy compatibility notes provided.')
"
    
    if [ $? -eq 0 ]; then
        log_success "Installation test passed!"
    else
        log_error "Installation test failed. Check the output above for details."
        exit 1
    fi
}

# Create project structure
create_structure() {
    log_info "Creating universal project structure..."
    
    # Core directories
    mkdir -p AI/{core,paraphraser,classifier,reasoning}
    mkdir -p {logs,model_cache,tests}
    
    # Create __init__.py files
    touch AI/__init__.py
    touch AI/{core,paraphraser,classifier,reasoning}/__init__.py
    
    log_success "Project structure created"
}

# Main setup function
main() {
    echo "================================"
    log_feature "Universal AI Paraphraser Setup"
    echo "================================"
    
    # Run setup steps
    setup_venv
    create_requirements
    install_dependencies
    test_installation
    create_structure
    
    # Final success message
    echo ""
    echo "================================"
    log_success " Universal Setup Complete!"
    echo "================================"
    
    log_info "✅ Compatible with ALL systems:"
    log_info "   - Windows (any version)"
    log_info "   - Linux (Ubuntu, CentOS, etc.)"  
    log_info "   - macOS (Intel + Apple Silicon)"
    log_info "   - Cloud servers (AWS, Google, Azure, Hetzner)"
    
    echo ""
    log_feature " Ready to run "

    
    echo ""
    log_feature "🌐 To start the service:"
    log_info "   python -m uvicorn AI.paraphraser.service:app --host 0.0.0.0 --port 8000"
    
    echo ""
    log_warning " First request may take 30-60s for model loading"
    log_info " Subsequent requests will be much faster"
    echo ""
    log_success " Your universal AI system is ready! "
}

# Error handling
trap 'log_error "Setup failed at line $LINENO"' ERR

# Run main setup
main "$@"