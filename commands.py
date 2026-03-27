python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"


python run_ingestion.py