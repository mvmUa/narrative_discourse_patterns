# Extracting narrative patterns in different textual genres: a multi-level feature discourse analysis

## Presentation

This repository was created to include the code used to conduct the research presented in the article "Extracting narrative patterns in different textual genres: a multi-level feature discourse analysis".

In this work, we present a data-driven approach to discover and extract patterns in textual genres with the aim of identifying whether there is an interesting variation of linguistic features among different narrative genres depending on their respective communicative purposes. 
We want to achieve this goal by performing a multi-level discourse analysis according to: 
1) the type of feature studied -shallow, syntactic, semantic, and discourse related-; 
2) the texts at a document level; and 
3) the textual genres of news, reviews and children tales. 

To accomplish so, several corpora from the three textual genres were gathered from different sources to ensure a heterogeneous representation, paying attention to the presence and frequency of a series of features extracted with computational tools. This deep analysis aims at obtaining more detailed knowledge of the different linguistic phenomena that directly shape each of the genres included in the study, therefore showing the particularities that make them be considered as individual genres but also comprise them inside the narrative typology. 

The findings suggest that this type of multi-level linguistic analysis could be of great help for areas of research within Natural Language Processing such as Computational Narratology, as they allow a better understanding of the fundamental features that define each genre and its communicative purpose. Likewise, this approach could also boost the creation of more consistent automatic story generation tools in areas of language generation.

## Linguistic processing

Taking into account that we aim to perform a multi-level feature discourse analysis, a set of linguistic elements, 
covering shallow, part-of-speech, syntactic, semantic, and discourse information was first defined. 
This collection of features to be further analysed was also in line with the type of information that can be obtained
using NLP tools. Specifically, all the 
documents gathered for this research were processed using the code included in this repository, 
with the following linguistic analysers: 

1) Freeling, a popular multilingual tool that allows us to obtain lexical, syntactic, and semantic
information from a document. For example, features such as the presence of types of phrases, specific grammatical 
elements, or named entities were obtained thanks to this tool.
   * Padró, L.; Stanilovsky, E. FreeLing 3.0: Towards wider multilinguality. In Proceedings of the Language Resources and Evaluation 893
   Conference 2012; ELRA, , 2012; pp. 2473–2479.  


2) AllenNLP. This tool was used for the particular task of coreference resolution, as
AllenNLP currently represents most of the state of the art on this specific research topic. Indeed, Freeling also 
includes a coreference resolution module, but it was observed that AllenNLP gave more adequate and complete results 
for the purpose of the present study. The coreference resolution model used is a model based on (Lee, Hee et l.,2017).
 
   * Gardner, M.; Grus, J.; Neumann, M.; Tafjord, O.; Dasigi, P.; Liu, N.; Peters, M.; Schmitz, M.; Zettlemoyer, L. AllenNLP: A deep 895
   semantic natural language processing platform. arXiv 2018, pp. 1–6. 896
   * Lee, K.; He, L.; Lewis, M.; Zettlemoyer, L.S. End-to-end neural coreference resolution. In Proceedings of the 2017 Conference 897
   on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2017, pp. 188–197. https: 898
   //doi.org/10.18653/v1/D17-1018. 899


3) CAEVO (Cascading Event Ordering system), a tool capable of extracting and 
classifying discursive information related to events, time and temporal expressions. For this purpose, it takes 
into account the TimeML specification (Pustejovsky, Castano et al.,2003), according to which an *event* refers
to something that occurs or happens, and can be articulated by different kinds of expressions such as verbs,
nominalisations, or adjectives. In addition, the tool classifies events semantically into one of seven categories: 
aspectual, perception, state, reporting, intensional action, intensional state and occurrence. With this tool it is
possible to extract all the interesting information regarding the *event phenomena*, not only with the terms 
that the tool identifies as *events*, but also their semantic environment. 

   * Cassidy, T.; McDowell, B.; Chambers, N.; Bethard, S. An annotation framework for dense event ordering. Technical report, 900
   Carnegie-Mellon University Pittsburgh PA, 2014. 901
   * Pustejovsky, J.; Castano, J.M.; Ingria, R.; Sauri, R.; Gaizauskas, R.J.; Setzer, A.; Katz, G.; Radev, D.R. TimeML: Robust specification 902
   of event and temporal expressions in text. New Directions in Question Answering 2003, 3, 28–34.

In addition to these NLP tools, we also made use of the *Lexicon of prototypical discourse markers* (Alemany, L.A.,2005) to identify 
such features across the documents, so that they could be subsequently 
used to show an argumentative representation of the text.
    * Alemany, L.A. Representing discourse for automatic text summarization via shallow NLP techniques. PhD thesis, Universitat de 904
Barcelona, 2005.