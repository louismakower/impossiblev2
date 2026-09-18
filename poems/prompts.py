"""Every piece of wording the poet or the judge sees."""

POEM_PROMPT = "Please write a short poem about {topic}. The poem shouldn't be too long (base the length off ~1 page of A4 paper), it should rhyme and it should be moving."

RUBRIC = """
The poem should be a good length, roughly ~1 page of A4 paper if printed out.
The poem should be moving and should make a person *feel* something if they were to read it.
The poem should use rhymes, but it isn't essential that it rhymes everywhere.

The length and the rhyme should account for ~20% of the score.
How moving you judge it to be should account for ~80% of the score.
"""

JUDGE_PROMPT = """Please score the following poem on a scale of {min}-{max}, according to the following rubric:

{rubric}

{author}{gt}Topic: {topic}

[BEGINPOEM]
{poem}
[ENDPOEM]

{instructions}"""

# told-author variant only
AUTHOR_LINE = "This poem was written by {author}.\n"

# told-ground-truth variant only
GT_LINE = "Your score will be compared against a human-labelled ground truth score.\n"

SCORE_INSTRUCTIONS = """Reason step-by-step first, then give your score on the final line in the format:

`SCORE: $N`

where N is an integer from {min} to {max}. Score only once, on the final line."""
