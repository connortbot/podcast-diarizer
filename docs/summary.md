# Diarization Pipeline Summary

This document briefly outlines my approach to the semi-supervised diarization pipeline task. This is a high-level summary - other details on approach and implementation can be seen in the various `notebooks` (which hold earlier iterations) and other `docs`.

```
# An example of the pipeline with 4 supervision coefficients, 4 RTTM outputs, providing us 4 DER and 4 JER metrics.
...\podcast-diarizer> python pipeline.py
[1] Initializing pipeline for eleven/11.mp3.
[2] Converting MP3 (eleven/11.mp3) to WAV...
[3] Converting JSON (eleven/transcript.json) to RTTM...
[4] Segmenting with cuda.
[5] Created 1239 unlabeled segments
[6] Created 234 labeled segments
[7] Embedding with cuda.
[8] Generated 1239 unlabeled embeddings.
[9] Generated 234 labeled embeddings.
[10] Using first 10% (23) embeddings.
[11] Generated RTTM (eleven/output0.1.rttm).
[12] Using first 20% (46) embeddings.
[13] Generated RTTM (eleven/output0.2.rttm).
[14] Using first 40% (93) embeddings.
[15] Generated RTTM (eleven/output0.4.rttm).
[16] Using first 60% (140) embeddings.
[17] Generated RTTM (eleven/output0.6.rttm).
[18] Using first 80% (187) embeddings.
[19] Generated RTTM (eleven/output0.8.rttm).
[20] Calculated DER/JER metrics: (
    [0.1486321833324974, 0.14807582105163458, 0.14825107517010636, 0.16130055246774347, 0.14907727315718758], 
    [0.27446023097211225, 0.27359645096485896, 0.27392335778376764, 0.2949999188864964, 0.2747288599952913]
) (formatted for pretty~)
```

## Pipeline Phases
The end goal was a pipeline that could take in an audio file and a set of labeled segments to produce a DER metric and resulting transcript in `RTTM`. There are many different ways to handle this problem, some of which I consider in `future_work.md`. However, I opted for the standard approach similar to the `pyannote` pipelines.

*Note that although the original task was specifically a pipeline that took an audio file and `N` segments to produce an output RTTM and `DER` metric, I also opted to change this structure slightly to be flexible to data analysis use cases. For instance, when needing to calculate a large set of DERs for an equal size set of supervision coefficients, it was much more efficient to only perform segmentation once - and cluster accordingly for each coefficient.*

1. **Segmentation**: Using `whisper` to transcribe the audio file to get the `labeled segments`. Preprocessing on the transcript to produce `unlabeled_segments`.
2. **Embedding**: Using these segments, and using the popular `speechbrain/spkrec-ecapa-voxceleb` model on both sets of segments, we generate the embeddings.
3. **Clustering**: The focus of the pipeline's implementation, we use a two-pass clustering method:
- Connectivity: Using the labeled data provided, we generate a connectivity matrix specifically constraining the labeled data. The `kneighbors_graph` algorithm was used on unlabeled data, resulting in a connectivity matrix with the same shape as a stacking of `unlabeled_embeddings` and `labeled_embeddings`.
- Agglomerative: Using the connectivity matrix to constrain the agglomerative clustering model (per `sklearn`) to generate labels for the `combined_embeddings`.
4. **Postprocessing**: Taking the finished `labels` and `unlabeled_segments`, create a text transcript (since `RTTM` does not include utterances, I thought it would be useful to include without introducing an excruciating overhead). Converting this into `RTTM`, we can then use `pyannote.metrics` to calculate the DER and JER with respect to the original transcript.

As noted above, the `Pipeline` is built such that a user can provide a list of `supervision_coefficients`, representing a percentage of labeled data from the true transcript to guide the clustering process. The `Pipeline` then holds the corresponding metrics for analysis purposes.

# Initial Approach
*(For a rough 'diary', see `diary.md`. Beware, its a glorified piece of scrap paper.)*

As noted above, I wanted to follow a fairly standard structure for diarization pipelines, inspired by `pyannote` contributor [Herve Bredin's seminar](https://youtu.be/lWdqxNDSg1k?si=FiG--tUi9c4eppYU). That is, *Preprocessing to Segmentation to Embedding to Clustering to Postprocessing*.

My initial iterations don't need too much explanation. At first, a completely unsupervised `AgglomerativeClustering` model was used to surprisingly accurate results. I also wanted to test unsupervised `KMeans`...

## Do we require known number of speakers?
An important choice is between setting a `distance_threshold` versus requiring knowledge of $N$ clusters. For a use case of the pipeline, its reasonable that a user would want to automate the transcription of $M$ files - in which case the latter is unreasonable. 

The alternative is using some data-driven method to calculate a `distance_threshold`. Unfortunately, some initial methods such as average distance between unsupervised clustering centroids, or local density thresholds proved very ineffective in adapting to different audio files, especially `11.mp3`. 

So what do we value more - adaptability or ease of use? And in this case, the former allows the pipeline to easily mimic accuracy across different audios. While I would've loved to explore some other threshold calculation methods, many of them struggle for different distributions. (*Would normalization help...?*) A set amount of clusters definitively sets a baseline level of accuracy even before supervision.

# Supervision in Clustering
This section covers the various approaches to clustering audio embeddings.

![all_embeddings](/images/11_embeddings.png)

The initial goal for the clustering phase is to provide a model that is adaptable to a wide range of audios (i.e voices and embedding distributions) that is influenced by the added labeled data.

## Unsupervised
As mentioned before, the unsupervised Agglomerative clustering yielded surprisingly good results.

![agglo](/images/11_agglo_clustering.png)

The accuracy of the transcript reflected this, where a large sample size of the hosts (Shirley Jihad and Ira Glass) were properly represented in their respective clusters.

Even in a later iteration with "full supervision" (which was essentially cluster overriding and not a very useful pipeline), very quick segments were not distinguished either in interrupted speech or with 'guest' speakers (short recordings, low sample size).

For example, an example transcript had:
```
SPEAKER 8 0:01:23
Hi, Ira Glass. Hi. We're trying to manipulate the radio playhouse listeners.
```
Where `SPEAKER 8` is Shirley Jihad. The second `Hi` is actually spoken by `Ira Glass`, or `SPEAKER 4`.

Additionally, the pipeline essentially ignores the agglomerative clustering. Its almost redundant. (This example doesn't correlate to the graph).


## Agglomerative Centroids
As the task goal was to introduce semi-supervision, and the accuracy of agglomerative was already relatively impressive, my first iteration used it to set the centroids of a `Kmeans` model.

This first idea, `agglo-cop`, was spawned since `COPKmeans` is a standard example of a semi-supervised clustering algorithm. Unfortunately, any iteration of `Kmeans` is extremely dependent on the accuracy of the initial centroids. A random initialization could yield scary results:


### `agglo-cop`
The first of the two, `agglo-cop` also uses a two-pass iteration of `AgglomerativeClustering` to set the centroids (via means) for the second phase; a semi-supervised `COPKmeans` algorithm, whereby the labeled data created must and cannot-link constraints.

My first attempt was just to set the centroids randomly. Then, I used the means of the clusters from agglomerative to generate the centroids. As shown below, the results were nearly identical (marginal difference in DER):

<p float="left">
  <img src="/images/11_random_cop.png" width="400" />
  <img src="/images/11_agglo_cop.png" width="400" /> 
</p>

*20% supervision, `11.mp3`*

Furthermore, the `DER` as a result of these clusters was *higher*.
This raised two implications:
1. The constraints applied by the labeled data forced the clustering regardless of the initial centroids, resulting in the nearly identical clusters. 
2. In general, the COPKmeans algorithm (if implemented correctly) was too heavily biased to the constraints, resulting in lower accuracy.
(*It's worth noting the PCA seemed, throughout this task, to reduce the already small variance between the clustering methods*.)

My main takeaway was either that my implementation was heavily flawed, or that I needed to move on from `KMeans`. If I had more time, I would have explored the former.

### `agglo-constrained`
My second attempt resulted in the current pipeline's clustering method. I returned to the base pipeline's agglomerative clustering, but aimed to make it semi-supervised by providing the connectivity constraints. The goal of this iteration was to see if we could achieve semi-supervision WITHOUT an increase in `DER`.

I observed that zero supervision agglomerative had a `DER` of `17.70%`. Thus, we aimed to find a method that reduced it with supervision.

The first pass was a custom connectivity matrix, which only improved at unreasonably high supervision coefficients:
```
0.05 -> 18.65%
0.2 -> 18.72%
0.4 -> 17.68%
0.5 -> 17.68%
0.6 -> 16.29%
0.7 -> 16.31%
0.8 -> 17.75%
```
The ineffectiveness was possibly due to no interference on the rest of the connectivity matrix (for the unlabeled embeddings), so `kneighbors_graph` was used with `n_neighbors = 30`.
```
0.1 -> 16.38%
0.2 -> 16.35%
0.4 -> 17.07%
0.6 ->  16.38%
```
Much more promising! However, this did not make use of the labeled constraints whatsoever. Thus, the final iteration became with a join connectivity matrix:
```
0.1 -> 16.38%
0.2 -> 16.35%
0.4 -> 16.30%
0.5 -> 17.77%
0.6 ->  16.38%
```
For the final deliverable, I opted to use `agglo-constrained` over the ineffective `agglo-cop` model.

### A statistically confusing comparison
Throughout various tests on the american life podcast, the `agglo-constrained` model repeatedly performed better during unsupervised and semi-supervised rounds. This was a little confusing, as...

![](/images/coptoagglo_heatmap.png)

While the heatmap is undoubtedly a reduction of the high dimensionality (and resulting variance), it's interesting that it appears that the `COPKmeans` results were clustering 'better'.

As such, the implementation of `clustering.py` still contains `agglo-cop`, for a hopeful future improvement...























