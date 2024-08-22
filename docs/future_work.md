# Future Work

## Alternative Clustering Methods
Despite the pipeline achieving a better `DER` with supervision and generally a mediocre level of quality, the amount of improvement yielded by supervision is disappointing. Many times, it felt like unsupervised clustering (common in some other pipelines such as `pyannote`'s) was already a baseline that barely moved with supervision. On top of the rest of the pipeline infrastructure, I spent a lot of time having to try new clustering methods only for the over-biasing of the labeled data to overtake any marginal benefits. 
A reasonable assumption now is that many simplistic clustering methods that are not at least guided in some non-algorithmic way (i.e neural networks) fail to capture the high dimension variance in the embeddings.

With some more time, I would explore Bayesian hierarchical methods (based on the effectiveness of heiarchical Agglomerative), Hidden Markov Models, or Gaussian Mixture Models. 

## Precise Segmentation and Embedding
An overlooked part of this pipeline is the attention to the pre-clustering phases. For a pipeline that could be specialized to podcasts, a fine-tuning of the embedding models or segmentation models may have greatly increased the variance, allowing for accurate clustering. Using the American Life podcast as a starting training set may have proved very effective...

With some more time, I would explore semi-supervised training via fine-tuning of the `VoxCeleb Speechbrain` model and, if possible, on `whisper`. Furthermore, segmentation alternatives such as extracting MFCCs with `librosa` or using `pyannote`'s would have been beneficial for comparison. 

## Pretrained pipelines
An entirely different way to approach the problem - there are many prebuild pipelines that offer fine-tuning. With some more time, it would be interesting to see how a pipeline pretrained exclusively on a use case like film scenes, podcasts, or action sequences would fare. 

## Data-driven thresholds
In the future, I would explore data-driven approaches to automatically determine critical parameters such as `n_neighbors` in `kneighbors_graph` or the `distance_threshold` in agglomerative clustering. These parameters are crucial for the success of the clustering process, but they currently require manual tuning, which can be impractical. (This would also remove the user requirement of knowing the amount of speakers!)

There are endless papers on data-driven methods. For instance, local density estimation or silhouette analysis (both proving very, very unhelpful during the task but could be improved).