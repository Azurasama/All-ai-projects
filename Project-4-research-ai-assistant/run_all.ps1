python scripts/download_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/prepare_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/build_index.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

streamlit run app.py
