# Combination of Real Data

## Data Downloading
you can use the following command to download real data from cloud:
`tosutil cp -r tos://cytoderm-embodied-ai/datasets/frankanav_part1/ep_xxxxx/ /home/datasets_v2/chenmao/frankanav/`, where `xxxxx` should be replaced from `00000` to `00099` (100 episodes).
The first episode (00000) has been downloaded. You can check its data format and information.

## Target
Conduct training using both the simulated benchmark `tsn-1k-var` and also the real data.

## Steps
1) Data download and process: the real data provides views of three cameras, but we only need the wrist camera (front). so when downloading, please only download what we need. you may first check the data format of the first downloaded episode. And maybe after downloading real data, you can apply data processing, like resolution compression if needed, and transform data format to the one like benchmark data. 
2) Update training code: the training uses the combination of both benchmark data and real data.
3) After training, conduct the closed-loop evaluation on the benchmark only, and see if it brings better performance.

You can now first download 3 episodes, process them, and i will review them.