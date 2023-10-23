# daisyRec recsys model

rm -rf src/model/daisyRec
rm -f requirements.txt

# clone to src/daisyRec
git clone -n --depth=1 --filter=tree:0 https://github.com/AmazingDD/daisyRec src/model/daisyRec

# sparse checkout model and utils folders and requirements
cd src/model/daisyRec
git sparse-checkout set --no-cone daisy/model daisy/utils daisy/assets requirements.txt
git checkout
touch "__init__.py"
cp requirements.txt ./../../../requirements.txt

cd ../../..

source patch.sh

echo -n "\nrequests>=2.30.0\n" >> requirements.txt
