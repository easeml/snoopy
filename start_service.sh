pip install fastapi # (python>=3.6)
pip install uvicorn # (python>=3.6)

#export SNOOPY_LIMIT_CPU=y
uvicorn service:app --reload --host 0.0.0.0
