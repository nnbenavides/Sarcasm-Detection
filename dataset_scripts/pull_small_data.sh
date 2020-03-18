# Code liberally inspired by and lifted from:
# https://github.com/kolchinski/reddit-sarc

mkdir -p SARC/2.0/pol
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/train-balanced.csv.bz2
bzip2 -d comments.json.bz2
bzip2 -d train-balanced.csv.bz2
mv comments.json SARC/2.0/pol/
mv train-balanced.csv SARC/2.0/pol/

mkdir ../static
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
rm wiki-news-300d-1M-subword.vec.zip
mv wiki-news-300d-1M-subword.vec ../static/

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
mkdir ../static/glove
mv glove*.txt ../static/glove/

mv SARC ../SARC