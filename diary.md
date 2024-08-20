# Diarizer Diary

# Initial Plan
For dev projects with a deadline, I personally enjoy separating all my work into X blocks, and allocating the blocks to days.

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

