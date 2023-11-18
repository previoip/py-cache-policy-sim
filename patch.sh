# daisyRec patch

echo "patching src/model/daisyRec"
cd src/model/daisyRec

echo "patching daisy"
cd daisy

echo "creating __init__.py file"
touch "__init__.py"

echo "patching model"
cd model

echo "creating __init__.py file"
touch "__init__.py"

echo "fixing rel imports"
find . -type f -exec sed -i 's/daisy\.model/src\.model\.daisyRec\.daisy\.model/g' {} \;
find . -type f -exec sed -i 's/daisy\.utils/src\.model\.daisyRec\.daisy\.utils/g' {} \;

# hotfix on mostpop model -> torch compat
echo "fixing torch compat"
find . -type f -exec sed -i 's/item_score\[cands_ids\]/item_score\[cands_ids\.long\(\)\]/g' {} \;

cd ../
echo "patching utils"
cd utils


echo "fixing deprecated lib mthods"
find . -type f -exec sed -i "s/\.iteritems/\.items/g" {} \;

echo "fixing funtion literals"
# hotfix utils functions var literals refers to default df column names
find . -type f -exec sed -i "s/row\['user'\]/row\['user_id'\]/g" {} \;
find . -type f -exec sed -i "s/row\['item'\]/row\['movie_id'\]/g" {} \;

echo "fixing rel imports"
find . -type f -exec sed -i 's/daisy\.utils/src\.model\.daisyRec\.daisy\.utils/g' {} \;


cd ../../../../../

echo "patching from patch files"

tempfolder=./patchtemp
if [ ! -d $tempfolder ]; then
    mkdir $tempfolder
fi


filename=""
pathname=""
patchdir="patches"
recpatch(){
    echo "patching models/$filename.py"
    if [ ! -f $tempfolder/$filename.py ]; then
        cp $pathname/$filename.py $tempfolder/$filename.py
    fi
    patch $tempfolder/$filename.py $patchdir/$filename.patch -o $pathname/$filename.py
    echo    
}

filename="sampler"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="EASERecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="Item2VecRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="KNNCFRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="LightGCNRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="NGCFRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="PopRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="PureSVDRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="VAECFRecommender"
pathname="src/model/daisyRec/daisy/model"
recpatch

filename="sampler"
pathname="src/model/daisyRec/daisy/utils"
recpatch

rm -rf $tempfolder