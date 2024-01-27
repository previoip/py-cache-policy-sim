set -b +e

source ./dev/dev_patch_vars.sh

src_path=./src/model/daisyRec/daisy
dst_path='./patchtemp'

if [ -d $dst_path ]; then
  rm -rf $dst_path
fi

mkdir $dst_path

reccp(){
  echo "copying" $1
  cp $1 $dst_path/$(basename "$1")
  dos2unix $dst_path/$(basename "$1")
}


reccp $src_path/$fp_AbstractRecommender
reccp $src_path/$fp_EASERecommender
reccp $src_path/$fp_FMRecommender
reccp $src_path/$fp_Item2VecRecommender
reccp $src_path/$fp_KNNCFRecommender
reccp $src_path/$fp_LightGCNRecommender
reccp $src_path/$fp_MFRecommender
reccp $src_path/$fp_NGCFRecommender
reccp $src_path/$fp_PopRecommender
reccp $src_path/$fp_PureSVDRecommender
reccp $src_path/$fp_NeuMFRecommender
reccp $src_path/$fp_NFMRecommender
reccp $src_path/$fp_VAECFRecommender
reccp $src_path/$fp_SLiMRecommender
reccp $src_path/$fp_sampler
reccp $src_path/$fp_utils

$SHELL