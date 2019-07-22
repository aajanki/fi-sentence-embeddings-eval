---
title: Evaluation tasks
section:
- title: Results
  href: results.html
- title: Embedding models
  href: models.html
- title: Evaluation tasks
  href: tasks.html
...

The [sentence classification models](models.html) in this study were
compared on the following Finnish sentence classification tasks:

**Eduskunta-VKK** [@eduskuntavkk]: The corpus is based on cabinet
ministers' answers to written questions by the members of Finnish
parliament.

The task is straight-forward sentence classification. Individual
sentences are extracted from the answer documents and the correct
class is the name of the ministry which answered the question. There
are 15 classes (ministries) and 49693 sentences as training data and
2000 sentences as test data. The evaluation measure is the F1 score.

Example sentences and classes:

| Sentence | Class |
| -------- | ----- |
| Lääkkeelliset kaasut kuten lääkehappi ovat lääkkeitä ( lääkelaki 395/1987 , 3 § ) . | sosiaali- ja terveysministeri |
| Kilpailutuskriteereissä noudatetaan lakia julkisista hankinnoista ja muita asiaan vaikuttavia säädöksiä . | sisäministeri |
| Poissaolot tulee selvittää ensisijaisesti opiston ja opiskelijan kesken . | sisäministeri |
| Tavoitteeksi asetettiin Porkkalan luonnonsuojelualueen perustaminen . | maatalous- ja ympäristöministeri |

**Opusparcus** [@creutz2018]: Sentence pair paraphrasing task. The
corpus is based on sentences from open movie subtitle datasets. The
size of the training dataset is about 20 million sentence pairs and
the testset is about 2000 pairs. The training data consists of
sentence pairs and a statistical estimate on how likely the two
sentences mean the same thing. The estimate is based on how often
sentences are aligned with similar translations over subtitles on
multiple languages. Model's task is to learn to predict this alignment
score.

The development and test sets consists of sentence pairs that have
been labeled by human annotators on the scale 1 (sentences do not mean
the same thing) to 4 (the sentences are paraphrases). The evaluation
measure is the correlation between model's prediction and the human
score. The model is evaluated only on the Finnish subset of the
corpus.

Example sentence pairs and their annotator scores on the test set:

| Sentence 1                            | Sentence 2                               | Annotator score |
| ------------------------------------- | ---------------------------------------- | :-------------: |
| Peruuttakaa täysillä .                | Täyttä taakse .                          | 3.5             |
| Ei minulla ole sijaa täällä .         | Ei täällä ole tilaa minulle .            | 4.0             |
| 12-14 tuntia .                        | Kerro se heille .                        | 1.0             |
| Nearyn perheellä ei ole sävelkorvaa . | Nearyn perheessä ei kukaan osaa laulaa . | 2.5             |
| Miltä hän näyttää ?                   | Minkänäköinen hän on ?                   | 4.0             |

**TDT categories** [@pyysalo2015]: The dataset contains Finnish
sentences extracted from many sources: blogs, Wikinews, Europarl,
student magazine articles, etc. The task is to predict the source of a
sentence. There are 8 classes, and about 8000 training and 1000 test
sentences. The evaluation measure is the F1 score.

Example sentences and their classes:

| Sentence | Class |
| -------- | ----- |
| Jäällä kävely avaa aina hauskoja ja erikoisia näkökulmia kaupunkiin. | Blog |
| Harvoin sitä on niin iloinen ihan vaan Oivariinin ja Oltermannin näkemisestä :D | Blog |
| Arvoisa puhemies, vapaat ja rehelliset vaalit eivät valitettavasti ole itsestäänselvyys vielä läheskään kaikissa maissa | Europarl |
| Sen vuoksi valiokunta päätti esittää tarkistuksia, jotka estävät tällaisen rehujen palauttamisen | Europarl |
| Tätä mantraa on toisteltu matalien korkojen aikoina tiuhaan. | Taloussanomat |
| Käytännössähän tässä rajoitetaan kansallisvaltioiden suvereniteettia talouspolitiikan puolella, Aunesluoma toteaa. | Taloussanomat |
| Saarten korkein kohta, 383 metriä, sijaitsee Uotsurilla. | Wikipedia |
| Moottoripyöriä valmistava osa onnistui vakuuttamaan sijoittajat ja teki hienon paluun huipulle. | Wikipedia |

**Ylilauta** [@ylilautacorpus]: Sentences extracted from Ylilauta
discussion forum. The task is to predict if a given pair of sentences
are consecutive sentences in the original text or not. The size of the
training dataset is 5000 sentence pairs and the test dataset is 2000
pairs. Half of the sentence pairs are in reality consecutive and half
are randomly paired sentences, which are assumed to be unrelated. The
evaluation measure is the classification accuracy.

Example sentence pairs:

| Sentence 1 | Sentence 2 | Class |
| ---------- | ---------- | ----- |
| Ja niin tässä on käynyt . | Jätkä on ostanut todella tökerösti tehdyn väärennöksen eikä suostu uskomaan sitä . | Consecutive |
| Lisää inkivääri ja valkosipuli , kypsennä vielä minuutti ja mausta . | Laitoin alogasgyselyyn 1. toiveeksi tulenjohdon . | Not consecutive |
| Tiedätkö miksi yritykset jakavat osinkoa ? | Miksi sijoitat juuri yhdysvaltojen yrityksiin ? | Consecutive |
| Eli toisinsanoen ainoa ' oikea ' taloustieto on se virallinen selitys mitä jossain tekstikirjassa lukee ? | Eli pähkinänkuoressa : filosofia toimii sinun tässä määrittelemäsi oikean tavan mukaan . | Not consecutive |
| Törmäsin koiran kanssa metsäreissulla pieneen hirsiseen metsästysmökkiin joka sijaitsi keskellä ei-mitään . | Sinne ei vienyt edes selvää polkua , mutta mökin pihassa näkyi kuitenkin asumisen merkkejä tältäkin syksyltä . | Consecutive |
| Harmi , että omat lupaavat junnut myydään jo näin aikaisessa vaiheessa . | Lisättäköön vielä , että viime torstainakin olisi ollut mainoksia jaettavana , mutta silloin olin onneksi vapaalla . | Not consecutive |

## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
