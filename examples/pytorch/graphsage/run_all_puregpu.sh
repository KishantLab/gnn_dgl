# ./run_script_gespmm.sh reddit 100
# ./run_script_gespmm.sh ogbn-arxiv 100
# ./run_script_gespmm.sh ogbn-products 100
# ./run_script_respmm.sh reddit 100
# ./run_script_respmm.sh ogbn-arxiv 100
# ./run_script_respmm.sh ogbn-products 100
# ./run_script_base.sh reddit 100
# ./run_script_base.sh ogbn-arxiv 100
# ./run_script_base.sh ogbn-products 100

# ./run_script_gespmm.sh igb-tiny 100
# ./run_script_gespmm.sh igb-medium 100
# ./run_script_gespmm.sh igb-large 100
# ./run_script_respmm.sh igb-tiny 100
# ./run_script_respmm.sh igb-medium 100
# ./run_script_respmm.sh igb-large 100
# ./run_script_base.sh igb-tiny 100
# ./run_script_base.sh igb-medium 100
# ./run_script_base.sh igb-large 100

# ./run_script_gespmm.sh igb-medium 10
# ./run_script_respmm.sh igb-medium 10
# ./run_script_base.sh igb-medium 10
# ./run_script_base.sh igb-large 10
# ./run_script_gespmm.sh igb-large 10
# ./run_script_respmm.sh igb-large 10

# ./run_script_gespmm.sh igb-small 10
# ./run_script_respmm.sh igb-small 10
# ./run_script_base.sh igb-small 10

# ./run_script_gespmm.sh amazon-products 10
# ./run_script_respmm.sh amazon-products 10
# ./run_script_base.sh amazon-products 10

# ./run_script_gespmm.sh wiki5M 10
# ./run_script_respmm.sh wiki5M 10
# ./run_script_base.sh wiki5M 10

# ./run_script_respmm_puregpu.sh ogbn-arxiv 100 20 4
./run_script_gespmm_puregpu.sh ogbn-arxiv 1
./run_script_base_puregpu.sh ogbn-arxiv 1

# ./run_script_respmm_puregpu.sh ogbn-arxiv 100 20 5
./run_script_gespmm_puregpu.sh ogbn-arxiv 1
./run_script_base_puregpu.sh ogbn-arxiv 1

# ./run_script_respmm_puregpu.sh reddit 100 20 4
# ./run_script_respmm_puregpu.sh reddit 100 20 5
# ./run_script_respmm_puregpu.sh reddit 100 30

./run_script_gespmm_puregpu.sh reddit 1
./run_script_gespmm_puregpu.sh reddit 1
# ./run_script_gespmm_puregpu.sh reddit 100 30

./run_script_base_puregpu.sh reddit 1
./run_script_base_puregpu.sh reddit 1
# ./run_script_base_puregpu.sh reddit 100 30


# ./run_script_respmm_puregpu.sh ogbn-products 100 20 4
./run_script_gespmm_puregpu.sh ogbn-products 1
./run_script_base_puregpu.sh ogbn-products 1
#
# ./run_script_respmm_puregpu.sh ogbn-products 100 20 5
./run_script_gespmm_puregpu.sh ogbn-products 1
./run_script_base_puregpu.sh ogbn-products 1

# ./run_script_respmm_puregpu.sh igb-small 100 20 4
./run_script_gespmm_puregpu.sh igb-small 1
./run_script_base_puregpu.sh igb-small 1

# ./run_script_respmm_puregpu.sh igb-small 100 20 5
./run_script_gespmm_puregpu.sh igb-small 1
./run_script_base_puregpu.sh igb-small 1


# ./run_script_respmm_puregpu.sh yelp 100 20 4
./run_script_gespmm_puregpu.sh yelp 1
./run_script_base_puregpu.sh yelp 1

# ./run_script_respmm_puregpu.sh yelp 100 20 5
./run_script_gespmm_puregpu.sh yelp 1
./run_script_base_puregpu.sh yelp 1


# ./run_script_respmm_puregpu.sh igb-small 100
# ./run_script_gespmm_puregpu.sh igb-small 100
# ./run_script_base_puregpu.sh igb-small 100

# ./run_script_gespmm_puregpu.sh wiki5M 100
# ./run_script_base_puregpu.sh wiki5M 100

# ./run_script_respmm_puregpu.sh wiki5M 100

