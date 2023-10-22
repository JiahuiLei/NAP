declare -a NameArray=(
    "K_8_cate_all_gt"
    "K_8_cate_all_v6.1_5455"
    "K_8_cate_all_v6.1_5455_retrieval"
)

for i in "${!NameArray[@]}"; do
    NAME=${NameArray[i]}
    echo $NAME
    python sample_pcl.py --src "../log/test/G/${NAME}" --dst "../log/test/PCL/${NAME}" --n_pcl 4096 --n_states 30
done