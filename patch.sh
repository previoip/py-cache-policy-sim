# daisyRec patch

cd src/model/daisyRec
cd daisy
touch "__init__.py"

cd model
touch "__init__.py"
find . -type f -exec sed -i 's/daisy\.model/src\.model\.daisyRec\.daisy\.model/g' {} \;
find . -type f -exec sed -i 's/daisy\.utils/src\.model\.daisyRec\.daisy\.utils/g' {} \;

# hotfix on mostpop model -> torch compat
find . -type f -exec sed -i 's/item_score\[cands_ids\]/item_score\[cands_ids\.long\(\)\]/g' {} \;

cd ../
cd utils

# hotfix utils functions var literals refers to default df column names
find . -type f -exec sed -i "s/row\['user'\]/row\['user_id'\]/g" {} \;
find . -type f -exec sed -i "s/row\['item'\]/row\['movie_id'\]/g" {} \;

# hotfix utils rel imports
find . -type f -exec sed -i 's/daisy\.utils/src\.model\.daisyRec\.daisy\.utils/g' {} \;
