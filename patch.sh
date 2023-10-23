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
# hotfix utils rel imports
find . -type f -exec sed -i 's/daisy\.utils/src\.model\.daisyRec\.daisy\.utils/g' {} \;

cd ../../../../../

echo "patching from patch files"

tempfolder=./patchtemp
if [ ! -d $tempfolder ]; then
    mkdir $tempfolder
fi

echo "patching Item2VecRecommender.py"
if [ ! -f $tempfolder/Item2VecRecommender.py ]; then
    cp src/model/daisyRec/daisy/model/Item2VecRecommender.py $tempfolder/Item2VecRecommender.py
fi
patch $tempfolder/Item2VecRecommender.py patches/item2vec.patch -o src/model/daisyRec/daisy/model/Item2VecRecommender.py
