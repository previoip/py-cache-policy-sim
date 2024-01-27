source ./dev/dev_patch_vars.sh

if [ -d $patch_path ]; then
  rm -rf $patch_path
fi
mkdir $patch_path

recdiff(){
  echo "building patch for" $4
  dos2unix $2/$4
  dos2unix $3/$(basename $4)
  diff $2/$4 $3/$(basename $4) > $1/$(basename $4 .py).patch 
  echo "done."
}

recdiff $patch_path $src_path $dst_path $fp_AbstractRecommender
recdiff $patch_path $src_path $dst_path $fp_EASERecommender
recdiff $patch_path $src_path $dst_path $fp_FMRecommender
recdiff $patch_path $src_path $dst_path $fp_Item2VecRecommender
recdiff $patch_path $src_path $dst_path $fp_KNNCFRecommender
recdiff $patch_path $src_path $dst_path $fp_LightGCNRecommender
recdiff $patch_path $src_path $dst_path $fp_MFRecommender
recdiff $patch_path $src_path $dst_path $fp_NGCFRecommender
recdiff $patch_path $src_path $dst_path $fp_PopRecommender
recdiff $patch_path $src_path $dst_path $fp_PureSVDRecommender
recdiff $patch_path $src_path $dst_path $fp_NeuMFRecommender
recdiff $patch_path $src_path $dst_path $fp_NFMRecommender
recdiff $patch_path $src_path $dst_path $fp_VAECFRecommender
recdiff $patch_path $src_path $dst_path $fp_SLiMRecommender
recdiff $patch_path $src_path $dst_path $fp_sampler
recdiff $patch_path $src_path $dst_path $fp_utils

$SHELL
