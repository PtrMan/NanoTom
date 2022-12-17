# utilities for text manipulation

# concatenates newlines of they dont end a sentence
#
# is useful to "repair" text from everything else than wikipedia!
def joinSentences(a):
    lines = a.split('\n')

    ### algorithm to concatenate text which is not ended by a dot(sentence end).
    # this is useful to clean up text of papers etc.

    lastLine = None

    acc = '' # accumulator

    for iLine in lines:
        lastLineHasDotAtEnd = False
        if lastLine is not None:
            if len(lastLine) > 0:
                lastLineHasDotAtEnd = lastLine[len(lastLine)-1] == '.'

        lastLine = iLine

        if lastLineHasDotAtEnd:
            acc = acc+'\n'+iLine
        else:
            acc = acc+' '+iLine

    return acc

