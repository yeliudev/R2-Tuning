# Download datasets from huggingface hub
base_url=https://huggingface.co/yeliudev/R2-Tuning/resolve/main/datasets

# Create the dataset folder
mkdir data && cd data

# QVHighlights
echo -e "\e[1;36mDownloading:\e[0m qvhighlights"
mkdir qvhighlights && cd qvhighlights
wget $base_url/qvhighlights/qvhighlights_train.jsonl
wget $base_url/qvhighlights/qvhighlights_val.jsonl
wget $base_url/qvhighlights/qvhighlights_test.jsonl
wget $base_url/qvhighlights/qvhighlights_clip_b32_vid_k4.tar.gz.aa
wget $base_url/qvhighlights/qvhighlights_clip_b32_vid_k4.tar.gz.ab
wget $base_url/qvhighlights/qvhighlights_clip_b32_vid_k4.tar.gz.ac
wget $base_url/qvhighlights/qvhighlights_clip_b32_vid_k4.tar.gz.ad
wget $base_url/qvhighlights/qvhighlights_clip_b32_vid_k4.tar.gz.ae
wget $base_url/qvhighlights/qvhighlights_clip_b32_txt_k4.tar.gz
cat qvhighlights_clip_b32_vid_k4.tar.gz.* | tar zxvf -
tar -zxvf qvhighlights_clip_b32_txt_k4.tar.gz
cd ..

# Ego4D-NLQ
echo -e "\e[1;36mDownloading:\e[0m ego4d"
mkdir ego4d && cd ego4d
wget $base_url/ego4d/nlq_train.jsonl
wget $base_url/ego4d/nlq_val.jsonl
wget $base_url/ego4d/ego4d_clip_b32_vid_k4.tar.gz.aa
wget $base_url/ego4d/ego4d_clip_b32_vid_k4.tar.gz.ab
wget $base_url/ego4d/ego4d_clip_b32_txt_k4.tar.gz
cat ego4d_clip_b32_vid_k4.tar.gz.* | tar zxvf -
tar -zxvf ego4d_clip_b32_txt_k4.tar.gz
cd ..

# Charades-STA
echo -e "\e[1;36mDownloading:\e[0m charades"
mkdir charades && cd charades
wget $base_url/charades/charades_train.jsonl
wget $base_url/charades/charades_test.jsonl
wget $base_url/charades/charades_clip_b32_vid_k4.tar.gz.aa
wget $base_url/charades/charades_clip_b32_vid_k4.tar.gz.ab
wget $base_url/charades/charades_clip_b32_txt_k4.tar.gz
cat charades_clip_b32_vid_k4.tar.gz.* | tar zxvf -
tar -zxvf charades_clip_b32_txt_k4.tar.gz
cd ..

# TACoS
echo -e "\e[1;36mDownloading:\e[0m tacos"
mkdir tacos && cd tacos
wget $base_url/tacos/train.jsonl
wget $base_url/tacos/val.jsonl
wget $base_url/tacos/test.jsonl
wget $base_url/tacos/tacos_clip_b32_vid_k4.tar.gz
wget $base_url/tacos/tacos_clip_b32_txt_k4.tar.gz
tar -zxvf tacos_clip_b32_vid_k4.tar.gz
tar -zxvf tacos_clip_b32_txt_k4.tar.gz
cd ..

# YouTube Highlights
echo -e "\e[1;36mDownloading:\e[0m youtube"
mkdir youtube && cd youtube
wget $base_url/youtube/youtube_anno.json
wget $base_url/youtube/youtube_clip_b32_vid_k4.tar.gz
wget $base_url/youtube/youtube_clip_b32_txt_k4.tar.gz
tar -zxvf youtube_clip_b32_vid_k4.tar.gz
tar -zxvf youtube_clip_b32_txt_k4.tar.gz
cd ..

# TVSum
echo -e "\e[1;36mDownloading:\e[0m tvsum"
mkdir tvsum && cd tvsum
wget $base_url/tvsum/tvsum_anno.json
wget $base_url/tvsum/tvsum_clip_b32_vid_k4.tar.gz
wget $base_url/tvsum/tvsum_clip_b32_txt_k4.tar.gz
tar -zxvf tvsum_clip_b32_vid_k4.tar.gz
tar -zxvf tvsum_clip_b32_txt_k4.tar.gz
cd ../..
