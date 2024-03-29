cd /data/kishan/gnn_dgl/
bash /data/kishan/gnn_dgl/script/build_dgl.sh -g
cd /data/kishan/gnn_dgl/python/
python3 setup.py install
python3 setup.py build_ext --inplace
