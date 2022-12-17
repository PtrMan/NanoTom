#### What is my purpose?

This is a Q&A system which can deal with natural language both for sources and questions.

#### Pipeline

**frontend** <br />
The frontend takes raw text (for example from wikipedia or pdf files) and converts them using a LM to a more semi-formal representation useful for representing facts as natural language sentences.

**backend** <br />
The backend tries to answer posed questions by searching for relevant sentences in the database and answering the question with these using a LM.
