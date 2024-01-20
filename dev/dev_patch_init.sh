set -b +e

src_path=./src/model/daisyRec/daisy
dst_path='./difftemp'

if [ -d $dst_path ]; then
  rm -rf $dst_path
  mkdir $dst_path
else
  mkdir $dst_path
fi

reccp(){
  echo "copying" $1
  cp $1 $dst_path/$(basename "$1")
}


reccp $src_path/$fp_AbstractRecommender
reccp $src_path/$fp_EASERecommender
reccp $src_path/$fp_Item2VecRecommender
reccp $src_path/$fp_KNNCFRecommender
reccp $src_path/$fp_LightGCNRecommender
reccp $src_path/$fp_NGCFRecommender
reccp $src_path/$fp_PopRecommender
reccp $src_path/$fp_PureSVDRecommender
reccp $src_path/$fp_VAECFRecommender
reccp $src_path/$fp_sampler

$SHELL