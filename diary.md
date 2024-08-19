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