# Results

Here, I show the results of the pipeline:
- Speed
- DER Accuracy
- JER Accuracy

# Hardware
CPU: `i7-11800H @ 2.30GHz` (misc.)
GPU: `NVIDIA GeForce RTX 2070` (Laptop) (Segmentation, Embedding)
RAM: `16GB`

# 11.mp3
Generating 100 transcripts for 100 supervision coefficients (0-1, with 0.01 step).
Segmentation: `2m 32.6s`
Total Time: `4m 19.7s`

# Error Rate
![](/images/metrics.png)
Clustering supervision failed to improve accuracy as supervision increased, but it noticeably reduced below the unsupervised rate of $17.70\%$.

# Best Transcript
### Best DER (14.80%)
[Best DER was at 10% supervision.](/eleven/der_best_0.1.txt). 
[RTTM version.](/eleven/der_best_0.1.rttm)
Excerpt:
```
SPEAKER 3 0:00:00
I'll pour this pestle on his ear, so will I make the net that will entail them all. It's an adult, Yago, who says that in Othello. And it's grown-ups that Machiavelli was writing about. When he wrote the prints, his book about manipulating others and seizing power. Notice he titled the book The Prince, not The Little Prince. The Little Prince is actually by somebody else. If you don't know that. But in our American lives, the real era of intrigue and manipulation for most of us is not adulthood. It's adolescence. When our social circle is at its most, constricting. Today on our program, a story of betrayal. And of someone who holds David Kuresh like powers over others. And who is only in the seventh grade. From WB Easy in Chicago. It's your radio playhouse. I'm Ira Glass. But before we get into the body of our story, we will try as adults to manipulate you a little bit. And put Central. Let's check in with Pledge Central. Shirley Jihad. 
SPEAKER 8 0:01:23
Hi, Ira Glass. Hi. We're trying to manipulate the radio playhouse listeners. 
SPEAKER 3 0:01:27
Well, I guess, manipulate has a tune of a negative connotation. 
SPEAKER 8 0:01:31
Oh, encourage, Kajol, Lure, maybe. 
SPEAKER 3 0:01:34
Yeah. 
SPEAKER 8 0:01:35
And we have all these entices. 
SPEAKER 3 0:01:36
How about entices? 
SPEAKER 8 0:01:37
Entices are very... 
SPEAKER 3 0:01:38
Seduce. Come on, baby. You're in a row. You're in such a row here. 
SPEAKER 4 0:01:43
Sure. 
```

### Best JER (26.69%)
[Best JER was at 71% supervision.](/eleven/jer_best_0.71.txt)
[RTTM version.](/eleven/jer_best_0.71.rttm)
Excerpt:
```
SPEAKER 1 0:00:00
I'll pour this pestle on his ear, so will I make the net that will entail them all. It's an adult, Yago, who says that in Othello. And it's grown-ups that Machiavelli was writing about. When he wrote the prints, his book about manipulating others and seizing power. Notice he titled the book The Prince, not The Little Prince. The Little Prince is actually by somebody else. If you don't know that. But in our American lives, the real era of intrigue and manipulation for most of us is not adulthood. It's adolescence. When our social circle is at its most, constricting. Today on our program, a story of betrayal. And of someone who holds David Kuresh like powers over others. And who is only in the seventh grade. From WB Easy in Chicago. It's your radio playhouse. I'm Ira Glass. But before we get into the body of our story, we will try as adults to manipulate you a little bit. And put Central. Let's check in with Pledge Central. Shirley Jihad. 
SPEAKER 4 0:01:23
Hi, Ira Glass. Hi. We're trying to manipulate the radio playhouse listeners. 
SPEAKER 1 0:01:27
Well, I guess, manipulate has a tune of a negative connotation. 
SPEAKER 4 0:01:31
Oh, encourage, Kajol, Lure, maybe. 
SPEAKER 1 0:01:34
Yeah. 
SPEAKER 4 0:01:35
And we have all these entices. 
SPEAKER 1 0:01:36
How about entices? Entices are very... Seduce. Come on, baby. You're in a row. You're in such a row here. 
SPEAKER 3 0:01:43
Sure. 
```

