git clone -b ${GIT_TAG} ${REPO_URL}
python3 -m pip install -r /app/${GIT_DIR}/requirements.txt
python3 /app/${GIT_DIR}/app.py