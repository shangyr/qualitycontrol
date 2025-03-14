wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

rm torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
rm torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
rm torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
rm torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

pip install torch-geometric==2.2.0