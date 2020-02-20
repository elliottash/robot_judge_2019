**Building a Robot Judge**

**Data Science for the Law**

**ETH Zurich -- Spring Term 2019**

**Course Syllabus**

Instructor: [Elliott Ash](http://elliottash.com/)

Contact: <ashe@ethz.ch>

Meeting times: Mondays, 1:15pm-3pm

Location:
[LFW](http://www.mapsearch.ethz.ch/map/mapSearchPre.do?gebaeudeMap=LFW&geschossMap=C&raumMap=5&farbcode=c010&lang=de)
C5

Course GitHub: <https://github.com/ellliottt/robot_judge_2019>

Office hours: After lecture or by appointment

**Course Summary**

Is a concept of justice what truly separates man from machine? Recent advances
in data science challenge us to reconsider this question. With expanding
digitization of legal data and corpora, alongside rapid developments in natural
language processing and machine learning, the prospect arises for automating
legal decisions.

Data science technologies have the potential to improve legal decisions by
making them more efficient and consistent. On the other hand, there are serious
risks that automated systems could replicate or amplify existing legal biases
and rigidities. Given the stakes, these technologies force us to think carefully
about notions of fairness and justice and how they should be applied.

This course introduces students to the data science tools that are unlocking
legal materials for computational and scientific analysis. We begin with the
problem of representing laws as data, with a review of techniques for
featurizing texts, extracting legal information, and representing documents as
vectors. We explore methods for measuring document similarity, clustering
documents based on legal topics or other features. Visualization methods include
word clouds and t-SNE plots for spatial relations between documents.

Law is embedded in language. Starting with case facts, we use natural language
processing to represent fact patterns as narrative sequences. This starts with
entity extraction and coreference resolution. Those entities are associated with
actions (as subject or object) and attributes (including associations with other
entities). Legal reasoning, in turn, can be modeled as sequences of norms and
causal statements.

We next consider legal prediction problems. Given the evidence and briefs in
this case, how will a judge probably decide? How likely is a criminal defendant
to commit another crime? How much additional revenue will this new tax law
collect? Students will investigate and implement the relevant machine learning
tools for making these types of predictions, including regression,
classification, and deep neural networks models.

We then use these predictions to better understand the operation of the legal
system. Under what conditions do judges tend to make errors? Against which types
of defendants do parole boards exhibit bias? Which jurisdictions have the most
tax loopholes? Students will be introduced to emerging applied research in this
vein. In a semester paper, students (individually or in groups) will conceive
and implement their own law-and-data-science research project.

Some programming experience in Python is required, and some experience with text
mining is highly recommended.

**Readings**

A lot of material will be skipped. These can be used as references along with
the content included in the slides.

-   Albadie and Cattaneo, “[Econometric Methods for Program
    Evaluation](https://www.dropbox.com/s/ji392gpyt4n009j/Abadie-Cattaneo-annurev-economics.pdf?dl=1)”
    (2018).

-   *Natural Language Processing in Python*, Third Edition, available at
    [nltk.org/book](https://www.nltk.org/book/).

-   Aurelien Geron*, Hands-On Machine Learning with Scikit-Learn & TensorFlow*,
    O’Reilly 2017 ([link](http://shop.oreilly.com/product/0636920052289.do)).

    -   [Jupyter notebooks for Geron’s
        book](https://github.com/ageron/handson-ml).

    -   The second half of the book is on deep learning. Course examples will
        use Keras (rather than TensorFlow), so skip the code examples if
        preferred.

    -   For R, see [Hands-on Machine Learning with R
        (beta)](https://bradleyboehmke.github.io/HOML/)

-   [Google Developers Text Classification
    Guide](https://developers.google.com/machine-learning/guides/text-classification/)

    -   This guide contains some practical tips and code examples for using text
        data.

-   Christopher Molnar, [Interpretable Machine
    Learning](https://christophm.github.io/interpretable-ml-book/)

-   Yoav Goldberg, [Neural Network Methods for Natural Language
    Processing](https://piazza-resources.s3.amazonaws.com/iyaaxqe1yxg7fm/iybxmq5nkds6ln/goldberg_2017_book_draft_20170123.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAR6AWVCBXSXECOGHS%2F20180911%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20180911T110633Z&X-Amz-Expires=10800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=FQoGZXIvYXdzEPP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDELWKZrf6Nt23r%2Fw4yK3AxykoidPEgIdpB2%2BXdhb7gFVRBuKmW1M%2BgfFnRVUsAQMPgsstnzSSpE1PtuYoHgArIJOAei5VuTWE9fyXLk%2Fxxqc7uRsdqAc7d7f6I0LQQCmP70bUqobbwdWcPzjOxdv1bOPnC5%2BdatIYgZEWwCqgGeGJq%2BJqsvV%2FXE4pO8Qtny57vJ%2BC7LPILwiHlnFu%2BAZqxwnb5jv8bdN7ir99uliWewj1fEGQbC1M9yGVr7oGoX1r%2FapDHKRRLcEt%2BOVD5iK%2B2L6hjsGDwhF85wLfJ%2Fk3zo8VCcfRnlfl4Yv5U3csrMBEcgNikLeJosmKZZXhjhF9%2Bi%2Flvl%2FyCLu3ey%2Bi4KgQUv7Al%2BwhMY0%2B5QO3FAzkRmz2kz0RQNsNJEVLRhvQXfuujlN7ha4g7DZ51aHcyLerZjgi99JSw4TY%2B24Yb1%2BUpdJiFXt%2FLi04R2BU%2BpGOpm%2FZXUi10ZmwO1lUrDwqaRhQTfDoORxD3StijxZxOPlfzKUFjszcCyqxSScabmx9HIOUgfaRiNmPJ3hM59d5209KU%2FoGaoGHizBi6OlfbzF%2F7HW0OtcuifYwGzRzMOAmFgJEc4rCaZu3D8okZ7e3AU%3D&X-Amz-Signature=89ac215243882c5d90217d29a7a9b6821bff28bb6a468382533da0f86d071310)
    (2017)

    -   This is a nice introduction to neural networks, with the advantage that
        it focuses on NLP problems.

-   Francois Chollet, *Deep Learning with Python* (2017).

    -   This is the only decent book on Keras.

-   [Papers with Code (NLP)](https://paperswithcode.com/area/nlp)

**Programming**

Python is the best option for text data and machine learning, used by most data
scientists. A teaching assistant will set up a private github repo for your
coursework.

[Python Setup
Instructions](https://docs.google.com/document/d/1UkCytHT4ZF-rDoh_buH6xb9mLz4GcGjT1qIu-pEWThI/edit?usp=sharing)

[Codecademy Online Python Course](https://www.codecademy.com/learn/learn-python)

**Assessment**

**Problem Sets (40%).** There will be homework assignments asking students to
implement the methods from the course on a provided practice corpus. Four
homeworks, at ten points each. Please submit as a Jupyter Notebook, uploaded to
your coursework repo. Two points deducted for each day late.

**Course Project (60%).** The main product is a short research paper, done
individually or in small groups. Students will come up with a research design in
consultation with the instructor or a project advisor. Students will give a
short presentation about their project at the end of the course ([presentation
guidelines](https://docs.google.com/document/d/16ocTdlPmqBug1pEKMq1rdDRBHbBqWI4_agJA5VJhfrI/edit?usp=sharing)).
Students can earn 2 additional credit points by signing up for a more
substantial project.

[Link to Project
Requirements](https://docs.google.com/document/d/1ntBfOA7G56l9gfLt9fxymcnb6U66_cXzYOzRX4Jp6X0/edit?usp=sharing)

**Acknowledgements**

Thanks to Chris Bail, Daniel L. Chen, Michael McMahon, Piero Molino, Arthur
Spirling, and Brandon Stewart for useful slide decks, on which some of these
lectures are based.

Course Outline
--------------

**Week 01: February 18**

1.  Course Overview

    -   Ash, “[Judge, Jury, and EXEcute File: The brave new world of legal
        automation](http://www.smf.co.uk/wp-content/uploads/2018/06/CAGE-FINAL-VF.pdf),”
        *Social Market Foundation*.

    -   Ash, “[Emerging Tools for a Driverless Legal
        System](https://www.dropbox.com/s/udet3zz7as2086m/Ash-JITE-Proof.pdf?raw=1),”
        *Journal of Institutional and Theoretical Economics*.

**Week 02: February 25**

1.  Machine Learning Essentials

    -   Methods:

        -   Geron, Chapter 2

        -   Geron, Appendix B

        -   Christodoulou et al, [A systematic review shows no performance
            benefit of machine learning over logistic regression for clinical
            prediction
            models](https://www.sciencedirect.com/science/article/pii/S0895435618310813)

    -   Applications:

        -   Dunn, Sagun, Sirin, and Chen, “[Early predictability of asylum court
            decisions](https://users.nber.org/~dlchen/papers/Early_Predictability_of_Asylum_Court_Decisions.pdf).”

        -   Bansak, Ferwerda, Hainmueller, Dillon, Hangartner, Lawrence, and
            Weinstein, [Improving refugee integration through data-driven
            algorithmic
            assignment](http://science.sciencemag.org/content/359/6373/325).

2.  Text Data Essentials

    -   Methods:

        -   Gentzkow, Kelly, and Taddy, “[Text as
            Data](https://web.stanford.edu/~gentzkow/research/text-as-data.pdf).”

        -   NLTK book, Chapters 1, 2, 4

    -   Applications:

        -   Ash and MacLeod, “Elections as selection and incentive device: The
            case of state supreme courts.”

        -   Ash and MacLeod, “Aging and retirement in a high-skill group: The
            case of state supreme court judges.”

        -   Ash, Chen, and Naidu (2018), “[Ideas have consequences: The effect
            of law and economics on American
            justice](http://elliottash.com/wp-content/uploads/2018/08/ash-chen-naidu-2018-07-15.pdf).”

        -   Klingenstein, Hitchcock, and DeDeo, “[The civilizing process in
            London’s old bailey](http://www.pnas.org/content/111/26/9419)”

**Week 03: March 4**

1.  N-Gram Models

    -   Methods:

        -   NLTK book, Chapter 3, 5, 7, 8.5

        -   Denny and Spirling (2018), “[Text Preprocessing for Unsupervised
            Learning: Why It Matters, When It Misleads, and What to Do about
            It](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2849145),”
            *Political Analysis*.

        -   Sebastian Raschka, “[Turn your Twitter Timeline into a Word
            Cloud](http://sebastianraschka.com/Articles/2014_twitter_wordcloud.html)”.

        -   Leon Derczynski, “Collocations,”
            <http://www.derczynski.com/sheffield/teaching/inno/7c.pdf>

    -   Applications:

        -   Chen et al, [Genealogy of
            Ideology](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2816707)

2.  Document Distance

    -   Methods:

        -   Leet et al, [An empirical evaluation of models of text document
            similarity](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.111.7144&rep=rep1&type=pdf).

        -   Brandon Rose, “[Document clustering in
            python](http://brandonrose.org/clustering).”

    -   Applications:

        -   Ganglmair and Wardlaw, [Complexity, Standardization, and the Design
            of Loan
            Agreements](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2952567).

        -   Abrahamson and Barber, [The evolution of national
            constitutions](https://www.dropbox.com/s/xt2sib3gybk9aqs/QJPS_Published.pdf?raw=1).

        -   Ash and Marian, “[The Making of International Tax Law: Evidence from
            Treaty
            Text](http://elliottash.com/wp-content/uploads/2019/01/Ash-Marian-NTJ-Submission.pdf)”.

        -   Kelly, Papanikolau, Seru, and Taddy, [Measuring technological
            innovation over the very long
            run](https://www.nber.org/papers/w25266).

        -   Hoberg and Phillips, [Text-based network industries and endogenous
            product
            differentiation](http://faculty.tuck.dartmouth.edu/images/uploads/faculty/gordon-phillips/text-based-network-industries-endogenous-product-differentiation.pdf).

        -   Ash and Labzina, Fox News Distorts Political Discourse

*March 5th, 4pm-6pm*

*IFW E42*

*Methods Seminar and Office Hours with Dr. Elena Labzina*

**Week 04: March 11**

1.  Regression, Classification, and Regularization

    -   Methods:

        -   Lavanya Shukla, [How I made top 0.3% on a Kaggle
            competition](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition).

        -   Geron, Chapter 3, 4, 7, and 8

        -   NLTK book, chapter 6

        -   Sokolova and Lapalme, [A systematic analysis of performance measures
            for classification
            tasks](http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf).

        -   Maitra and Yan, [PCA and PLS for
            Regression](https://www.casact.org/pubs/dpp/dpp08/08dpp76.pdf)

        -   Lee and Verleysen, [On the role of metaparameters in t-distributed
            Stochastic Neighbor
            Embedding](https://pdfs.semanticscholar.org/presentation/8258/cd8ed624bcaad3f5e75570086ac641e260e0.pdf)

        -   [Feature Selection in
            Scikit-Learn](http://scikit-learn.org/stable/modules/feature_selection.html)

        -   Manish Pathak, [Using XGBoost in
            Python](https://www.datacamp.com/community/tutorials/xgboost-in-python)

        -   Zhang and Hancock, [A graph-based approach to feature
            selection](https://link.springer.com/chapter/10.1007/978-3-642-20844-7_21).

        -   [Temporal
            Cross-Validation](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/3_modeling_and_machine_learning/temporal-cross-validation)

    -   Applications:

        -   Ash, Morelli, and Osnabrugge, “A policy topic model for political
            texts: Method and an application to New Zealand’s electoral reform.”

        -   Cao, Ash, and Chen, “Automated Fact-Value Distinction in Legal
            Language.”

        -   Ash, [What drives partisan tax policy? The effective tax
            code](http://elliottash.com/wp-content/uploads/2018/12/tax-laws-draft-current.pdf).

        -   Katz, Bommarito, and Blackman, “[A general approach for predicting
            the behavior of the Supreme Court of the United
            States](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174698).”

        -   Mainali, Meier, Ash, and Chen, “[Automated Classification of Modes
            of Moral Reasoning in Judicial
            Decisions](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3205286).”

**Week 05: March 18**

1.  Topic Models

    -   Methods:

        -   Grimmer and Stewart, “[Text as Data: The Promise and Pitfalls of
            Automatic Content Analysis Methods for Political
            Texts](https://web.stanford.edu/~jgrimmer/tad2.pdf).”

        -   Brandon Rose, “[Document clustering in
            python](http://brandonrose.org/clustering).”

        -   Shivam Bansal, “[Beginner's guide to topic modeling in
            Python](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/).”

        -   Blei, Ng and Jordan, 2003. “[Latent Dirichlet
            Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf),”
            *Journal of Machine Learning Research*.

        -   Roberts et al, “Structural Topic Models for Open-Ended Survey
            Responses”.

        -   Christian Fong and Justin Grimmer, “[Discovery of treatments from
            text corpora](https://stanford.edu/~jgrimmer/SE_Short.pdf).”

    -   Applications:

        -   Hansen, McMahon, and Prat, [Transparency and deliberation with the
            FOMC: A computational linguistics
            approach](https://academic.oup.com/qje/article/133/2/801/4582916).

        -   Draca and Schwarz, [How polarized are citizens? Measuring ideology
            from the ground
            up](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3154431).

2.  Deep Learning Essentials

    -   Geron, Chapter 10 and 11 ([Jupyter
        Notebook](https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb))

    -   [Keras Docs: Guide to the Sequential
        model](https://keras.io/getting-started/sequential-model-guide/)

    -   Axel Bellec, [POS tagging with
        Keras](https://becominghuman.ai/part-of-speech-tagging-tutorial-with-the-keras-deep-learning-library-d7f93fa05537)

    -   Karlijn Willems, [Keras Tutorial: Deep Learning in
        Python](https://www.datacamp.com/community/tutorials/deep-learning-python)

    -   Parneet Kaur, [Neural Networks in
        Keras](http://parneetk.github.io/blog/neural-networks-in-keras/)

    -   [Keras Docs: Intro to the Functional
        API](https://keras.io/getting-started/functional-api-guide/)

    -   Geron, Appendix D

    -   [Google Developers Text Classification Guide, Part
        2](https://developers.google.com/machine-learning/guides/text-classification/step-2-5)

    -   [AdaBound, An optimizer that trains as fast as Adam and as good as
        SGD.](https://github.com/Luolc/AdaBound)

*March 19th, 4pm-6pm, IFW E42*

*Methods Seminar and Office Hours with Dr. Elena Labzina*

*March 21st, Problem Set 1 Due (covering Lectures 1-6)*

**Week 06: March 25**

1.  Embedding Layers and Word Embeddings

    -   Methods:

        -   Guo and Berkahn, [Entity Embeddings of Categorical
            Variables](https://arxiv.org/pdf/1604.06737v1.pdf)

        -   Chollet, Chapter 6

        -   Yoav Goldberg and Omer Levy, “[Word2Vec explained: Deriving Mikolov
            et al's Negative Sampling Word Embedding
            Method](https://arxiv.org/pdf/1402.3722.pdf)”.

        -   Piero Molino, “[Word embeddings: Past, present, and
            future](http://w4nderlu.st/teaching/word-embeddings)”.

        -   Chip Huyen, [How to structure your model in
            TensorFlow](https://web.stanford.edu/class/cs20si/2017/lectures/notes_04.pdf)

        -   Matt Kusner, Yu Sun, Nicholas Kolkin, and Killian Weinberger, “[From
            word embeddings to document
            distances](https://mkusner.github.io/publications/WMD.pdf)”.

        -   Andy Thomas, [A Word2Vec Keras
            Tutorial](http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)

        -   Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, Andrej Risteski,
            [Linear Algebraic Structure of Word Senses, with Applications to
            Polysemy](https://arxiv.org/abs/1601.03764)

        -   Allen and Hospedales, [Analogies Explained: Towards Understanding
            Word Embeddings](https://arxiv.org/pdf/1901.09813.pdf)

        -   Peters, Ruder, and Smith, [To tune or not to tune: Adapting
            pretrained representations to diverse
            tasks](https://arxiv.org/abs/1903.05987).

        -   Peters et al, [Deep contextualized word
            representations](https://arxiv.org/abs/1802.05365).

        -   Antoniak and Mimno, [Evaluating the stability of embedding-based
            word
            similarities](https://mimno.infosci.cornell.edu/info3350/readings/antoniak.pdf).

    -   Applications:

        -   Rudolph and Blei, “[Dynamic Bernoulli Embedding Models for Language
            Evolution](https://arxiv.org/abs/1703.08052)”.

        -   Hamilton, Clark, Leskovec, and Jurafsky, 2016, [Inducing
            domain-specific sentiment lexicons from unlabeled
            corpora](https://arxiv.org/pdf/1606.02820.pdf).

        -   Rheauly, Beelen, Cochrane, and Hirst, [Measuring emotion in
            parliamentary debates with automated text
            analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168843#sec002).

        -   Caliskan et al, “[Semantics derived automatically from language
            corpora contain human-like
            biases](http://opus.bath.ac.uk/55288/4/CaliskanEtAl_authors_full.pdf)”

        -   Bolukbasi et al, [Man is to computer programmer as woman is to
            homemaker: Debiasing word
            embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf).

        -   Garg et al 2018, [Word embeddings quantify 100 years of gender and
            ethnic
            stereotypes](http://www.pnas.org/content/early/2018/03/30/1720347115/tab-figures-data)

        -   Kozlowski, Taddy, and Evans 2018, [The geometry of culture:
            Analyzing meaning through word
            embeddings](https://arxiv.org/abs/1803.09288)

        -   Ruiz, Athey, and Blei, “[SHOPPER: A Probabilistic Model of Consumer
            Choice with Substitutes and
            Complements](https://arxiv.org/abs/1711.03560)”

        -   Stoltz and Taylor, [Concept Mover's Distance: Measuring Concept
            Engagement in Texts via Word
            Embeddings](http://scholar.google.com/scholar_url?url=https://osf.io/5hc4z/download/%3Fformat%3Dpdf&hl=en&sa=X&d=15059126327028699503&scisig=AAGBfm1Iqk1Qlxgms3En0EfCxKT7U0kUPQ&nossl=1&oi=scholaralrt&hist=o5uDfHMAAAAJ:3575187946198448895:AAGBfm0j-py_zEiokxtBExrZAD91IPAdKQ)

**Week 07: April 1**

1.  Document Embeddings

    -   Methods:

        -   Le and Mikolov, “[Distributed representations of sentences and
            documents](https://arxiv.org/abs/1405.4053).”

        -   Dai, Olah, and Le, [Document Embedding with Paragraph
            Vectors](https://arxiv.org/pdf/1507.07998.pdf).

        -   Arora, Liang, and Ma, “[A simple but tough-to-beat baseline for
            sentence embeddings](https://openreview.net/pdf?id=SyK00v5xx).”

        -   Wu et al, S[tarspace: Embed all the
            things!](https://arxiv.org/abs/1709.03856)

        -   Bhatia, Lau, and Baldwin, “[Automatic labeling of topics with neural
            embeddings](https://arxiv.org/abs/1612.05340)”

        -   Trifonov, Ganea, Potapenko, and Hofmann, [Learning and evaluating
            sparse interpretable sentence
            embeddings](https://arxiv.org/abs/1809.08621)

    -   Applications:

        -   Ash and Chen 2018, [Mapping the geometry of law using document
            embeddings](http://elliottash.com/wp-content/uploads/2018/09/Mapping_the_geometry_of_law_using_document_embeddings.pdf).

        -   Ornaghi, Ash, and Chen 2019, Implicit associations in judicial
            language.

        -   Demsky et al, 2019, [Analyzing Polarization in Social Media: Method
            and Application to Tweets on 21 Mass
            Shootings](https://arxiv.org/abs/1904.01596)

        -   Gennaro, Ash, and Loewen, Emotional Language in Political Speech.

2.  Syntactic Parsers

    -   Methods:

        -   [ClearNLP Dependency
            Labels](https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md)

        -   Jurafsky and Martin, [Statistical
            Parsing](https://web.stanford.edu/~jurafsky/slp3/12.pdf).

        -   [spaCy 101](https://spacy.io/usage/spacy-101)

    -   Applications:

        -   Ash, MacLeod, and Naidu (2019), The language of contract: Promises
            and power in collective bargaining agreements

*April 8th: Holiday, no class*

*April 12th: Project Topic Due*

**Week 08: April 15**

1.  Machine Learning and Causal Inference (42 slides)

    -   Egami, Fong, Grimmer, Roberts, and Stewart, [How to Make Causal
        Inferences Using Texts](https://arxiv.org/pdf/1802.02163.pdf)

    -   Hartford, Lewis, Leyton-Brown, and Taddy, [Deep IV: A Flexible Approach
        for Counterfactual
        Prediction](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)

    -   Yixin Wang and David Blei, “[The Blessings of Multiple
        Causes](https://arxiv.org/abs/1805.06826).”

    -   Belloni, Chernozhukov, and Hansen, “[High-Dimensional Methods and
        Inference on Structural and Treatment
        Effects](https://www.aeaweb.org/articles?id=10.1257/jep.28.2.29)”

    -   Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins,
        [Double/Debiased Machine Learning for Treatment and Causal
        Parameters](https://arxiv.org/abs/1608.00060).

    -   Margaret Roberts, Brandon Stewart, and Richard Nielsen, “[Matching
        Methods for High-Dimensional Data with Applications to
        Text](http://www.margaretroberts.net/wp-content/uploads/2015/07/textmatching.pdf)”

    -   Online tool for Causal Graphs: <http://dagitty.net>

    -   [Nick Huntington-Klein Causality Lecture
        Slides](http://www.nickchk.com/econ305.html).

**Student presentations:**

-   Pascal Schärli

-   Eric Stavarache, Louis Abraham, Panayiotou Panayiotis, and Akram Yassir

*April 16th, 4pm-6pm, IFW E42*

*Methods Seminar and Office Hours with Dr. Elena Labzina*

*April 18, Problem Set 2 Due (covering Lectures 6-10)*

*April 22nd, Holiday*

**Week 09: April 29**

**\*Guest Lecture, Hoda Heidari\***

1.  Algorithms and Bias

    -   Ziyuan Zhong, [A tutorial on fairness in Machine
        Learning](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb)

    -   Kleinberg, Lakkaraju,Leskovec, Ludwig, and Mullainathan, [Human
        decisions and machine
        predictions](https://cs.stanford.edu/~jure/pubs/bail-qje17.pdf).

    -   Berk et al, Fairness in Criminal Justice Risk Assessments

    -   Sandra Mayson, [Bias In, Bias
        Out](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257004)

    -   Amaranto et al, Algorithms as Prosecutors

    -   Bower, Niss, Sun, and Vargo, [Debiasing representations by removing
        unwanted variation due to protected
        attributes](https://arxiv.org/abs/1807.00461Q).

    -   Sloan, Naufal, and Caspers, [The effect of risk assessment scores on
        judicial behavior and defendant
        outcomes](http://ftp.iza.org/dp11948.pdf).

    -   Kleinberg et al, [Algorithmic
        Fairness](https://www.cs.cornell.edu/home/kleinber/aer18-fairness.pdf).

    -   Kleinberg, Ludwig, Mullainathan, and Sunstein, [Discrimination in the
        age of algorithms](https://www.nber.org/papers/w25548.pdf).

**Student presentations:**

-   Yinghao Dai and Rok Sikonha

-   Yannic Kilcher, Paulina Grnarova, Florian Schmidt, and Kevin Roth

**Week 10: May 6**

1.  Measuring Polarization in Text

    -   Gentzkow, Shapiro, and Taddy, “[Measuring Group Differences in
        High-Dimensional Choices: Method and Application to Congressional
        Speech](https://www.brown.edu/Research/Shapiro/pdfs/politext.pdf).”

    -   Ash, Morelli, and Van Weelden. “[Elections and Divisiveness: Theory and
        Evidence](http://elliottash.com/wp-content/uploads/2013/08/posturing_online.pdf)”

    -   Ash, Chen, and Lu, “[Polarization of Precedent and Prose in U.S. Circuit
        Courts,
        1800-2013](https://users.nber.org/~dlchen/papers/Motivated_Reasoning_in_the_Field.pdf).”

    -   Demsky, Garg, Voigt, Zou, Gentzkow, Shapiro, and Jurafsky, [Analyzing
        polarization in social media: Method and application to Tweets on 21
        Mass Shootings ](https://arxiv.org/pdf/1904.01596.pdf)

**Student presentations:**

-   Simon Schaefer

-   Michal Sudwoj

-   Tanos Heman

-   Matthaus Hee

**Week 11: May 13**

1.  Measuring Entropy and Complexity in Text

    -   Simon DeDeo, [Information theory for intelligent
        people](http://tuvalu.santafe.edu/~simon/it.pdf)

    -   Ash, Morelli, and Vannoni.

    -   Benoit, Munger, and Spirling (2017), “[Measuring and Explaining
        Political Sophistication Through Textual
        Complexity](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3062061)”.

    -   Katz and Bommarito, “[Measuring the complexity of the law: The United
        States
        Code](https://link.springer.com/article/10.1007/s10506-014-9160-8).”

    -   Ganglmair and Wardlaw, [Complexity, Standardization, and the Design of
        Loan
        Agreements](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2952567).

    -   Ash and Guillot, Tax Law Complexity

**Student presentations:**

-   Gishor Sivanrupan

-   Tianqi Wang and Luyang Han

-   Dominik Borer

-   Tobias Luscher

-   Philip Nikolaus

-   Swagatam Sinha

-   Raphael Husistein and Andreas Bloch

*May 14, , 4pm-6pm, IFW E42*

*Methods Seminar and Office Hours with Dr. Elena Labzina*

*May 16, Problem Set 3 Due (covering Lectures 10-14)*

**Week 12: May 20**

1.  Model Interpretability

    -   Molnar, [Interpretable Machine
        Learning](https://christophm.github.io/interpretable-ml-book/) (2.2,
        2.6, 5)

    -   Sarkar, [Explainable artificial
        intelligence](https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476)

    -   Ribeiro, Singh, and Guestrub, [Local interpretable model-agnostic
        explanations (LIME): An
        introduction](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime).

    -   Lundberg and Lee, [A unified approach to interpreting model
        predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

    -   Lakkaraju et al, [Faithful and customizable explanations of black box
        models](https://cs.stanford.edu/people/jure/pubs/explanations-aies19.pdf).

    -   Dan Becker, [Permutation
        Importance](https://www.kaggle.com/dansbecker/permutation-importance)

    -   Lipton, [The mythos of model
        interpretability](https://arxiv.org/abs/1606.03490)

    -   [Python Notebook with Model Interpretation
        Examples](http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html#Permutation-Importance)

**Student presentations:**

-   Simon Huber

-   Peric Lazar

*May 25,* [Project
Proposals](https://docs.google.com/document/d/1ntBfOA7G56l9gfLt9fxymcnb6U66_cXzYOzRX4Jp6X0/edit?usp=sharing)
*Due*

**Week 13: May 27**

**Student presentations:**

-   Julian Merkofer

-   Imre Kertesz

-   Hussain Alqattan

-   Simon Zurfluh

-   Marc Bolliger

-   Stefan Walser

-   Noe Javet

-   Kieran Mepham

-   Moritz Halter

-   Roland Friedrich

*June 7-8: Conference on Data Science and Law*

*June 11, , 4pm-6pm, IFW E42*

*Methods Seminar and Office Hours with Dr. Elena Labzina*

*June 14: Problem Set 4 Due (covering Lectures 16-18)*

*July 15: Rough drafts due (optional for 3-credit students).*

*August 19: Final papers due.*

**Extra Readings**

1.  Deep Learning, Part II

    -   Josh Gordon, [Symbolic and Imperative APIs in TensorFlow
        2.0](https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021).

    -   Josh Tobin, [Troubleshooting Deep Neural
        Networks](http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf)

    -   Convolutional Neural Networks

        -   Geron Chapter 13

        -   Ujjwal Karn, [An intuitive explanation of Convolutional Neural
            Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

        -   Huotari, [Spelling out Convolution
            1D](http://www.jussihuotari.com/2017/12/20/spell-out-convolution-1d-in-cnns/)

        -   Johnson and Zhang, [Effective use of word order for text
            categorization with convolutional neural
            networks](https://arxiv.org/pdf/1412.1058.pdf)

        -   Zhang and LeCun, [Text understanding from
            scratch](https://arxiv.org/abs/1502.01710)

        -   Jason Brownie, [How to Develop an N-gram Multichannel Convolutional
            Neural Network for Sentiment
            Analysis](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/)

        -   Cholley, [How CNNS see the
            world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)

        -   Tal Perry, [Convolutional methods for
            text](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f)

    -   Recurrent Neural Networks

        -   Geron Chapter 14

        -   Andrej Karpathy, [The unreasonable effectiveness of recurrent neural
            networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

    -   Autoencoders

        -   Geron Chapter 15

        -   Chollet, [Building Autoencoders in
            Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

        -   Gabriel Goh, [Decoding the Thought
            Vector](http://gabgoh.github.io/ThoughtVectors/)

    -   Transformers:

        -   Devlin et al, [BERT: Pre-training of Deep Bidirectional Transformers
            for Language Understanding](https://arxiv.org/abs/1810.04805).

        -   Vaswani et al, [Attention is all you
            need](https://arxiv.org/abs/1706.03762)

        -   Radford et al, [Language models are unsupervised multitask
            learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

        -   [GPT-2
            Demo](https://colab.research.google.com/github/ak9250/gpt-2-colab/blob/master/GPT_2.ipynb)

        -   [Geron notebook on transformers in
            Keras](https://github.com/ageron/handson-ml2/blob/master/16_nlp_with_rnns_and_attention.ipynb)

    -   Applications

        -   Lei et al (2016), [Rationalizing neural
            predictions](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf).

        -   Roberts and Sanford (2018).

        -   Trifonov et al, Sparse interpretable sentence embeddings.

        -   Amini, Soleimany, Schwarting, Bhatia, and Rus, [Uncovering and
            Mitigating Algorithmic Bias through Learned Latent
            Structure](http://www.aies-conference.com/wp-content/papers/main/AIES-19_paper_220.pdf)

2.  Information Extraction and Narrative Schemas

    -   Surdeanu et al 2011, [Customizing an information extraction system for a
        new domain](https://nlp.stanford.edu/pubs/relms2011.pdf).

    -   Jurafsky and Chambers, [Unsupervised learning of narrative schemas and
        their participants](http://www.aclweb.org/anthology/P09-1068).

    -   Lehmann et al, [DBPedia -- A large-scale, multilingual knowledge base
        extracted from
        Wikipedia](https://www.researchgate.net/profile/Christian_Bizer/publication/259828897_DBpedia_-_A_Large-scale_Multilingual_Knowledge_Base_Extracted_from_Wikipedia/links/0deec52e78a6e95b73000000/DBpedia-A-Large-scale-Multilingual-Knowledge-Base-Extracted-from-Wikipedia.pdf).

    -   Wyner and Peters, [On rule extraction from
        regulations](http://wyner.info/research/Papers/WynerPetersJURIX2011.pdf).

    -   Mohammad and Turney, [Crowdsourcing a word-emotion association
        lexicon](https://arxiv.org/abs/1308.6297).

Graham Neubig’s [Neural Nets for NLP
Course](http://phontron.com/class/nn4nlp2017/schedule.html)

-   Chetty and Hendren, Machine Learning for Intergenerational Mobility

-   Leskovec, Jure and Backstrom, Lars and Kleinberg, Jon. 2009. Meme-tracking
    and the Dynamics of the News Cycle. Proceedings of the 15th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining.

-   Athey, Tibshirani, and Wager, “[Generalized random
    forests](https://arxiv.org/abs/1610.01271)”

-   Supervised learning:

    -   Kevin Markham, “A friendly introduction to linear regression using
        Python”,
        <https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb>

    -   Aarshay Jain, “A complete tutorial on ridge and lasso regression in
        Python,”
        <https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/>.

    -   J. Warmenhoven and R. Jordan Crouser, “PCR and PLS in Python”,
        <http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11/Lab%2011%20-%20PCR%20and%20PLS%20Regression%20in%20Python.pdf>

    -   Yhat, “Random forests in Python,”
        <http://blog.yhat.com/posts/python-random-forest.html>

    -   OPIG, “Using random forests in Python with Scikit-learn”,
        http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/

-   Text analysis with R:

    -   Text Mining with R, <http://tidytextmining.com/>

    -   Ken Benoit, “Getting started with quanteda”,
        https://cran.r-project.org/web/packages/quanteda/vignettes/quickstart.html

– Margaret Roberts, Brandon Stewart, and Dustin Tingley, “stm: R package for
Structual Topic Models”,
<https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf>

-   Erin Hengel (2017), “[Publishing While
    Female](http://www.erinhengel.com/research/publishing_female.pdf).”

– Trevor Hastie, Robert Tibshirani, and Jerome Friedman, Elements of Statistical
Learning,
http://www.statedu.ntu.edu.tw/bigdata/The%20Elements%20of%20Statistical%20Learning.pdf.

-   Methods:

    -   Raj Chetty, “[Sufficient statistics for welfare analysis: A bridge
        between structural and reduced-form
        methods](https://www.annualreviews.org/doi/10.1146/annurev.economics.050708.142910),”
        *Annual Review of Economics*.

    -   Piketty and Saez, “[Optimal Taxation of Top Labor Incomes: A Tale of
        Three
        Elasticities](http://www.ucl.ac.uk/~uctp39a/PikettySaezStantchevaAEJ2014.pdf)”
        *American Economic Journal: Economic Policy*.

-   Law and Political Economy

    -   Maskin and Tirole, “The politician and the judge: Accountability in
        government”, *American Economic Review* (2004)
        [[LINK](https://scholar.harvard.edu/files/maskin/files/the_politician_and_the_judge.pdf)]

    -   Alesina and Tabellini, “Bureaucrats or politicians?” *American Economic
        Review* (2007)
        [[LINK](https://scholar.harvard.edu/files/alesina/files/bureaucrats_or_politicians_part_1.pdf)]

    -   Gennaioli and Shleifer, [The evolution of common
        law](https://scholar.harvard.edu/files/shleifer/files/evolution_jpe_final.pdf).

• High-dimensional econometrics:

– Alexandre Belloni, Daniel Chen, Victor Chernozhukov, and Christian Hansen,
“Sparse Models and Methods for Optimal Instruments with an Application to
Eminent Domain”, https://arxiv.org/abs/1010.4345.

– Susan Athey and Guido Imbens, “Recursive partioning for heterogeneous causal
effects”, https://arxiv.org/abs/1504.01132.

3.4 Social-Science Applications

∗ Dictionary Methods:

∙ David Baron, Renee Bowen, and Salvatore Nunnari, “Durable coalitions and
communication: Public versus private negotiations,”
http://didattica.unibocconi.it/mypage/upload/53135_20170721_031738_BBN_DYNBARGCHAT.PDF

• Topic models:

– Kevin Quinn, Burt Monroe, Michael Colaresi, Michael Crespin, and Dragomir
Radev, “How to analyze political attention with minimal assumptions and costs,”
http://clair.si.umich.edu/\~radev/papers/AJPS2010.pdf.

• Supervised learning:

– Matt Gentzkow and Jesse Shapiro, “What drives media slant?”
