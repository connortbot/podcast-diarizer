# Diarizer Diary

# Initial Plan
For dev projects with a deadline, I personally enjoy separating all my work into X blocks, and allocating the blocks to days. A working pipeline needs to be completed within 2 days/ 48 hours, so I'll be working on this mainly from late Aug 19th to late Aug 21st.

### Block 1: Setup
- Dev env setup.
- Brief outline of overall pipeline.
- Brief outline of all the writeup sections.

### Block 2: Base Pipeline
- JSON to RTTM conversion for `transcript.json` and file conversion to `.wav` for `11.mp3`.
- Segementation w/ `whisper`.
- Segment embeddings.
- Clustering (try with K-Means vs Agglomerative?).
- Write transcript (can convert to RTTM, but this is really just a first pass to see how things go on `11`).
- Calculate DER (using `dscore`, which also has JER)?
- Writeup!

### Block 3: Refinement
- Use `transcript.json` for semi-supervision.
- Writeup!

### Block 4: Metrics
- Writeup! $\Rightarrow$ DER graph showing relationship between supervision and DER.
- Show metrics experimenting with clustering (if having the time).

### Block 5: Write, Write, Write
- Write pipeline overview, user instructions, results summary.
- Writeup for future work.
- Output most optimized transcript and convert to RTTM (i.e lowest DER).


# Block 1: Setup
Taking some inspiration from (Mark Padaka?)'s writeups, I'll have my :writeup" composed of:
- Folder of `notebooks` showing scrap work.
- The writeups in `Block 5`.
- Outside notebook for all figures in any of the writeups (`visuals.ipynb`)

### Outline for Overall Pipeline
1. **Preprocessing**: Convert test `transcript` to `RTTM` as needed, convert audio to `.wav` if/as needed.
2. **Segementation**: Initial segments with `whisper`, extract and output segment embeddings.
3. **Clustering**: Using method like Agglomerative, cluster segement embeddings. Create initial diarized transcript.
4. **Post processing**: Convert to `RTTM` and calculate DER/JER.

Note: unsure of how I plan on dealing with semi-supervision in context of everything else, will have to figure out later.


# Block 2: Base Pipeline
I'm off to go tap tap code :D
Added `conversion.ipynb` which covers just the transcript conversion. The file conversion to `.wav` would be added throughout adding segmentation.

Spent more time than I should've on redoing my Python envs - and apparently Whisper doesn't like NumPy 2.0.1...

## Transcription
Noted that `pyannote` is an alternative for transcription, or specifically segmentation to `whisper`. The API for `whisper` is very simple, so I have no problem with committing to it. However, with more time, I'd love to see how a fully `pyannote`'d pipeline would look...

The whisper transcription was done on an Intel Laptop. First time around was on the CPU, which is an i7-11800H. Will likely switch to GPU usage and keep it consistent for remaining tests.
- CPU-1: `9m 0.4s` (i7-11800H)
- GPU-1: `2m 39.5s` (RTX 3070)

## Segment Embeddings
I'll be using the pretrained speaker embedding model from `pyannote.audio`, `speechbrain/spkrec-ecapa-voxceleb`.

## Clustering
Used simple Agglomerative clustering on the embeddings. I want the pipeline to handle audio without being given the number of speakers - to use agglo, I need to provide a distance threshold. The first pass of these clusters is pretty inaccurate. Considering the segments themselves are already not guided by any supervision - the only way the transcript becomes somewhat accurate is with manual trial-and-error with the `distance_threshold`. (`1150` was more or less the most accurate for this simplistic pipeline with no supervision.)

# Block 3: Refinement
Just to get a working pipeline in, we'll have absolutely full supervision (which is obviously redundant) by using the transcript to influence the centroids of the clusters when we perform agglomerative. This means just reading from the `transcript.rttm` for segmentation, using the same model to create embeddings (while attaching speaker labels).

During clustering, we:
- Generate clusters without any supervision.
- Using the labels and labeled embeddings, calculate the mean of each distinct cluster according to the labels.
- For each unlabeled embedding, calculate the closest cluster and assign it to that cluster. It now has a set of refined labels.

This ended up working relatively flawlessly, especially when compared to the base pipeline. Of course, this is expected due to the literal answer sheet handed to the model, and it still makes common diarization mistakes during overlapping speech. (Even if the words themselves aren't overlapped, because the specificity of segments is still limited by the original `whisper` segmentation).

For example, the refined transcript has:
```
SPEAKER 8 0:01:23
Hi, Ira Glass. Hi. We're trying to manipulate the radio playhouse listeners.
```
Where `SPEAKER 8` is Shirley Jihad. The second `Hi` is actually spoken by `Ira Glass`, or `SPEAKER 4`.

Additionally, the pipeline essentially ignores the agglomerative clustering. Its almost redundant.

# Block 4: Metrics
On Day 1.
At this point in the process, there were other things I really wanted to work on that just was not time-efficient for my limit:
- Smaller segments to reduce mistakes with multiple speakers in one segment (see Block 3 above).
- Other methods of clustering such as K-Means or specifically [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0885230821000619?via%3Dihub). This can change the amount of input we would require from a pipeline user, such as number of speakers as opposed to a large excerpt of a transcript (the former of which is much easier to provide for most filmmakers). Additionally, other ways of influencing clusters with labels besides centroids.
- Increase overall speed - segmentation (~2.5m) is the most time-consuming aspect. Besides better hardware, I'd like to learn more about how...and possibly influencing segmentation by fine-tuning a model on the dataset.

For this block, though, I should focus on semi-supervision, that is, only providing percentages of the complete transcript. `metrics.ipynb` will calculate DERs between the true `transcript.rttm` and the other transcripts I've generated at each iteration of the pipeline to see how it improves. I'll also start showing the relationships between DER,JER and amount of supervision.

Allons-y:bangbang: :o

Update:
After working on it a bit more, I'm stupidly realizing that `refined-pipeline.ipynb`'s clustering section is super redundant - it literally overrides whatever work was done with Agglomerative and just assigns everything to the labels...

I've run into a fundamental issue that seems to be withstanding in constrained clustering - whereby the added constraints fail to provide any improvements or, in some cases, even weaken the clustering. A good example is adding constraint matrices to `scikit-learn` `AgglomerativeClustering`, which weakens the clustering by introducing weaknesses of single linkage. 

### agglo-cop
My first pass was using an iteration of `AgglomerativeClustering` to set the centroids (via means) for a semi-supervised `COPKmeans` algorithm, whereby the labeled data is stacked onto the unlabeled and used to create must-link and cannot-link constraints.
While this is the best hit at semi-supervision that I've found, it was simply worse the `base-pipeline`. The DER went from roughly `15%` to `17%`, indicating that the semi-supervision overly biased the latter clustering.

I'll attempt another method, which is doing one pass of `AgglomerativeClustering` with a connectivity constraint matrix.

### agglo-constrained
No supervision: `17.70%`
With supervision coeff (custom connectivity matrix):
`0.05 -> 18.65%`
`0.2 -> 18.72%`
`0.4 -> 17.68%`
`0.5 -> 17.68%`
`0.6 -> 16.29%`
`0.7 -> 16.31%`
`0.8 -> 17.75%`
Using `kneighbors_graph` with `n_neighbors = 30`:
`0.1 -> 16.38%`
`0.2 -> 16.35%`
`0.4 -> 17.07%`
`0.6 ->  16.38%`
Using both methods in tandem:
`0.1 -> 16.38%`
`0.2 -> 16.35%`
`0.4 -> 16.30%`
`0.5 -> 17.77%`
`0.6 ->  16.38%`


There are other options that I'd love to compare and contrast once I have the time:
- Cross validation, Bayesian (mentioned earlier)
- Models like GMMs or HMMs ([example](https://github.com/cr7anand/semi-speaker-diarization))

I was hoping that I'd have the time to explore other methods, especially neural-network based ones, that would reduce DER more with supervision. But, for now, I'll have to settle for this pipline's relatively accuracy (although 16% is still quite high...)


# Block 5: Write, Write, Write
At this point, I spent way too long on Block 4 - I think to get noticeably better results with the supervision I would need to spend some more time researching and implementing a specific method (e.g Bayesian HMM, as mentioned earlier). In which case, I should build the pipeline to be modular enough that I can switch out the clustering method at will.

Since Block 4 took so long, going to spend a little more time adapting everything into useable `PDPipeline` + any other needed classes. Will consolidate this diary writeup into `future_work.md`, `summary.md`, `results.md`. Finally, use the finished pipeline to make all output examples, with their respective transcripts labeled with their DER/JER metrics.

### Metrics
For the writeup, I'll include the following figures:
- DER/JER with respect to percentage of labeled segments provided
- DER/JER across each pipeline iteration, e.g COPKmeans vs KNeighbors+Agglomerative
- Performance on VoxConversev0.3 compared to pyannote/speaker-diarization
