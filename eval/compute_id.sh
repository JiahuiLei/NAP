REF="../log/test/PCL/K_8_cate_all_gt"
Ns=10
Np=2048

python instantiation_distance.py --gen $REF --ref $REF --n_states $Ns --n_pcl $Np

GEN="../log/test/PCL/K_8_cate_all_v6.1_5455"
python instantiation_distance.py --gen $GEN --ref $REF --n_states $Ns --n_pcl $Np
python instantiation_distance.py --gen $GEN --ref $GEN --n_states $Ns --n_pcl $Np
GEN="../log/test/PCL/K_8_cate_all_v6.1_5455_retrieval"
python instantiation_distance.py --gen $GEN --ref $REF --n_states $Ns --n_pcl $Np
python instantiation_distance.py --gen $GEN --ref $GEN --n_states $Ns --n_pcl $Np