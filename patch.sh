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
tempfolder=./patchtemp
recpatch(){
    echo "  >> $filename.py"
    if [ ! -f $tempfolder/$filename.py ]; then
        cp $pathname/$filename.py $tempfolder/$filename.py
    fi
    dos2unix $tempfolder/$filename.py
    patch $tempfolder/$filename.py $patchdir/$filename.patch -o $pathname/$filename.py
    echo
}

if [ ! -d $tempfolder ]; then
    mkdir $tempfolder
fi

echo
echo "! begin patching daisyRec !"

touch "src/model/daisyRec/__init__.py"
touch "src/model/daisyRec/daisy/__init__.py"

pathname="src/model/daisyRec/daisy/model"
echo "# target directory $pathname"

echo "fixing from patch files"

filename="EASERecommender"
recpatch

filename="FMRecommender"
recpatch

filename="Item2VecRecommender"
recpatch

filename="KNNCFRecommender"
recpatch

filename="LightGCNRecommender"
recpatch

filename="MFRecommender"
recpatch

filename="NGCFRecommender"
recpatch

filename="NFMRecommender"
recpatch

filename="NeuMFRecommender"
recpatch

filename="PopRecommender"
recpatch

filename="PureSVDRecommender"
recpatch

filename="VAECFRecommender"
recpatch

filename="SLiMRecommender"
recpatch

filename="AbstractRecommender"
recpatch

echo "fixing rel imports"
p1="from\ daisy\.model"
p2="from\ src\.model\.daisyRec\.daisy\.model"
recsed
p1="from\ daisy\.utils"
p2="from\ src\.model\.daisyRec\.daisy\.utils"
recsed

echo "fixing torch compat (mostpop)"
p1="item_score\[cands_ids\]"
p2="item_score\[cands_ids\.long\(\)\]"
recsed


pathname="src/model/daisyRec/daisy/utils"
echo "# target directory $pathname"

echo "fixing from patch files"

filename="sampler"
recpatch

filename="utils"
recpatch

echo "fixing deprecated lib methods"
p1="\.iteritems"
p2="\.items"
recsed


echo "fixing rel imports"
p1="daisy\.utils"
p2="src\.model\.daisyRec\.daisy\.utils"
recsed

# rm -rf $tempfolder

echo
echo "done"

$SHELL