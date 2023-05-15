conda create -n chess_data python=3.8
conda activate chess_data
pip install -r requirements.txt
python -m ipykernel install --user --name=chess_data
jupyter notebook --> kernel: chess_data
