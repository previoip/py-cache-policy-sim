# daisyRec patch

# func recsed
pathname=""
p1=""
p2=""
recsed(){
    echo "  >> sed $p1 --> $p2"
    find $pathname -type f -exec sed -i "s/${p1}/${p2}/g" {} \;
}

# func recpath
filename=""
patchdir="patches"
recpatch(){
    echo "  >> $filename.py"
    if [ ! -f $tempfolder/$filename.py ]; then
        cp $pathname/$filename.py $tempfolder/$filename.py
    fi
    patch $tempfolder/$filename.py $patchdir/$filename.patch -o $pathname/$filename.py
    echo
}

tempfolder=./patchtemp
if [ ! -d $tempfolder ]; then
    mkdir $tempfolder
fi

echo
echo "! begin patching daisyRec !"

touch "src/model/daisyRec/__init__.py"
touch "src/model/daisyRec/daisy/__init__.py"

pathname="src/model/daisyRec/daisy/model"
echo "# target directory $pathname"

echo "fixing rel imports"
p1="daisy\.model"
p2="src\.model\.daisyRec\.daisy\.model"
recsed
p1="daisy\.utils"
p2="src\.model\.daisyRec\.daisy\.utils"
recsed

echo "fixing torch compat (mostpop)"
p1="item_score\[cands_ids\]"
p2="item_score\[cands_ids\.long\(\)\]"
recsed

echo "fixing from patch files"

filename="EASERecommender"
recpatch

filename="Item2VecRecommender"
recpatch

filename="KNNCFRecommender"
recpatch

filename="LightGCNRecommender"
recpatch

filename="NGCFRecommender"
recpatch

filename="PopRecommender"
recpatch

filename="PureSVDRecommender"
recpatch

filename="VAECFRecommender"
recpatch


pathname="src/model/daisyRec/daisy/utils"
echo "# target directory $pathname"

echo "fixing deprecated lib methods"
p1="\.iteritems"
p2="\.items"
recsed

echo "fixing invalid funtion literals"
p1="row\['user'\]"
p2="row\['user_id'\]"
recsed
p1="row\['item'\]"
p2="row\['movie_id'\]"
recsed

echo "fixing rel imports"
p1="daisy\.utils"
p2="src\.model\.daisyRec\.daisy\.utils"
recsed

echo "fixing from patch files"

filename="sampler"
recpatch

# rm -rf $tempfolder

echo
echo "done"