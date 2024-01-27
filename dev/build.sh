source ./dev/vars.sh

if [ -d $patch_path ]; then
  rm -rf $patch_path
fi
mkdir $patch_path

recdiff(){
  echo "building patch for" $1
  dos2unix $tmp_path/$(basename $1)
  dos2unix $dif_path/$(basename $1)
  diff $tmp_path/$(basename $1) $dif_path/$(basename $1) > $patch_path/$(basename $1 .py).patch 
  echo "done."
}

recdiff $fp_AbstractRecommender
recdiff $fp_EASERecommender
recdiff $fp_FMRecommender
recdiff $fp_Item2VecRecommender
recdiff $fp_KNNCFRecommender
recdiff $fp_LightGCNRecommender
recdiff $fp_MFRecommender
recdiff $fp_NGCFRecommender
recdiff $fp_PopRecommender
recdiff $fp_PureSVDRecommender
recdiff $fp_NeuMFRecommender
recdiff $fp_NFMRecommender
recdiff $fp_VAECFRecommender
recdiff $fp_SLiMRecommender
recdiff $fp_sampler
recdiff $fp_utils

$SHELL
