# 1. Go to project root
cd ~/ToxicRadar

# 2. Create venv (only needed once)
python3 -m venv venv

# 3. Activate venv (needed every time)
source venv/bin/activate

# 4. Verify venv is active
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# 5. Run universal setup
chmod +x universal_startup.sh
./universal_startup.sh

# 6. Test everything works
python -c "
import torch
import transformers
print('✅ All packages installed in venv!')
"

# WHAT IF YOU RAN THE SCRIPT WRONG

# Clean up (remove incorrectly installed packages)
exit venv
rm -rf venv  # Remove the venv
pip3 uninstall torch transformers fastapi  # Remove from system (optional)

# Start over correctly:
python3 -m venv venv
source venv/bin/activate
./AI/universal_startup.sh