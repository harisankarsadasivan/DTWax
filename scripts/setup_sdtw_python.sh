echo -e "\n${G1}setting up virtual environment...${G2}"
python3.6 -m venv sf-venv3 && \
    source sf-venv3/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    ipython kernel install --name "sf-venv3" --user || \
    fail=true
if $fail; then echo -e "\n${R1}virtual environment setup failed${R2}"; exit 1; else echo -e "${G1}done!${G2}"; fi