## How to reproduce results 


1. CUDA_LAUNCH_BLOCKING=0 python3 gpu_rewt_ss_generic.py /tmp l1 0 l3 l4 0 l6 qg 5 <dataset_path> <num_class> nn 0 <batch_size> <lr_learning_rate> <gm_learning_rate> normal f1 

- <dataset_path> is the path to the directory of the stored LFs
- <num_class> is number of classes in the dataset (for eg, TREC has 6 classes and SMS has 2 classes)
- <batch_size> is kept sa 32 in all our experiments
- <lr_learning_rate> is set as 0.0003
- <gm_learning_rate> is set as 0.01
- last argument can be either f1 or accuracy where f1 refers to macro-F1.


### How to automatically generate LFs
1. cd reef/
2. python generate_human_lfs.py dataset(imdb/trec/sms/youtube) count/lemma savetype(dict/lemma)

- 1st argument is dataset name (i.e imdb/trec/sms/youtube/sst5/twitter)
- 2nd argument generation of raw (count) or lemmatized feature (lemma) 
- 3rd argument is path of the directory to save the generated LFs

### Generate LFs from snuba
1. cd reef/
2. python generic_generate_labels.py youtube normal dt 1 26 yt_val2.5_sup5_dt1 count

- 1st argument is dataset name (i.e imdb/trec/sms/youtube/sst5/twitter)
- 2nd argument is prefix of generated pkl files
- 3rd argument is number of LFs per step
- 4th argument is number of epochs
- 5th argument is storage path (LFs/data/youtube/<storage_path>) where pkl files will be stored
- 6th argument is type of features
