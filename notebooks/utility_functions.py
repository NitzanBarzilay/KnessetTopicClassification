import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.test.utils import datapath


def get_stopwords() -> set[str]:
    """
    Returns a list of stopwords in hebrew based on pre-loaded file along with added words that are unique to this corpus.
    """
    stop_path = "../data/1 - Original data/heb_stopwords.txt"
    with open(stop_path, encoding="utf-8") as in_file:
        lines = in_file.readlines()
        res = [l.strip() for l in lines]
        res.extend([",", ".",'-','–',"\"","\t","ה", "ל", "ב", "ו", "ש", "מ", "של", "על", "את", "או",
                         "הוא", "לא", "אם", "כל", "כ", "עם", "הם", "היא", "הן", "ח", "ראו",
                         "בעד", "נגד", "אינו", "אינה", "נוכח", "נוכחת", "בהתאם", "לסעיף", "לחוק-יסוד",
                        'כנסת', 'חברי','תודה', 'בבקשה', 'גברתי', 'כבוד',
                         'חוק', 'אדוני', 'הצעת', 'החוק', 'חברת', 'הממשלה', 'היושב-ראש,', 'שר', 'רבה',
                        'ועדת', 'תיקון', 'נמנעים', 'הצבעה', 'נא', 'הנושא','מס\'',
                    "יודעים", "רשות","דיון", "לכן", "לכולם",'הדיון','לדיון', 'בעניין','הוועדה', 'הזה',
                    'הישיבה', 'כולל', 'לשמוע', 'בדיון','הכנסת', 'מתכבד', 'מתכבדת', 'ישיבת', 'בעצם',
                    'לפתוח', 'דוח', 'בוועדה', 'תקנות','לענייני' ,'לעניין', 'פותח', 'בוועדת', 'המשנה',
                    'מס', 'נמצאת','מבקש','נמצאים','נמצא', 'נשמע', 'רואים','ממשלה','הדוח', 'המדינה',
                    'לצערי', 'תודה','שעבר', 'נכבדים', 'שאתם', 'במדינת','משרד', 'לומר', 'אתמול', 'לקריאה','כפי'
                       ,'סדר', 'תיקוני', 'בחוק','לשנות' ,'סעיף','ראשונה', 'ההצעה', 'בכנסת', 'אורחים', 'נכבדים',
                    'דובר','השעה', 'להציג', 'סדר-היום','בהצעת','נציגי','ואנחנו', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024',
                    'ידי','בברכה','כפי','שאתם','בקריאה','הערות','משרדי','לפרוטוקול'])
    return set(res)

def plot_top_n_words_hist(text, n, corpus_stop_set=None, plot_non_stopwords=False, print_non_stopwords=False) -> None:
    """
    Prints a bar chart of top n words in given text.
    """
    sns.set(rc={"figure.figsize": (6, 5)})
    corpus = [word.translate(str.maketrans('', '', '.,)(')) for quote in text.str.split().values.tolist() for word in quote]
    top = Counter(corpus).most_common()
    title = f'Top {n} words in corpus'
    x, y = zip(*top[:n])
    x = [word[::-1] for word in x]
    sns.barplot(x=list(y), y=x).set(xlabel='Frequency', ylabel='Word',title=title)
    plt.show()

    if plot_non_stopwords:
        title = f'Top {n} non-stop-words in corpus'
        x, y = [], []
        for word, count in top:
            if (word not in corpus_stop_set):
                x.append(word)
                y.append(count)
                if len(x) == n:
                    break
        if print_non_stopwords:
            print(x)
        x = [word[::-1] for word in x]
        sns.barplot(x=list(y), y=x).set(xlabel='Frequency', ylabel='Word',title=title)

def get_top_ngram(corpus, n=None):
    """
    Returns a list of all n-grams in given corpus, sortes by frequency.
    """
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = []
    for word, idx in vec.vocabulary_.items():
        words_freq.append((word, sum_words[0, idx]))
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq

def df_to_words_list(data):
    words_lists = [text.split() for text in data]
    words = []
    for words_list in words_lists:
        words.extend([w[::-1] for w in words_list])
    return words


def plot_top_n_bigrams_hist(text, stop_words, n, plot_non_stopwords=False):
    """
    Plot a histogram of the n most common non-stopwords in the df.
    """
    sns.set(rc={"figure.figsize": (6, 5)})
    text = df_to_words_list(text)
    title = f'Top {n} bigrams in corpus'
    top_bigrams = get_top_ngram(text, 2)
    print([(x[::-1],y) for x,y in top_bigrams])
    x, y = [], []
    for bigram, count in top_bigrams:
        word1, word2 = bigram.split(" ")
        if not word1.isnumeric() and not word2.isnumeric():
            x.append(bigram)
            y.append(count)
        if len(x) == n:
            break

    sns.barplot(x=y, y=x).set(xlabel='Frequency', ylabel='Bigram', title=title)
    plt.show()
    if plot_non_stopwords:
        title = f'Top {n} non-stop-words bigrams in corpus'
        x, y = [], []
        for bigram, count in top_bigrams:
            word1, word2 = bigram.split(" ")
            if (word1[::-1] not in stop_words and not word1.isnumeric()
                    and word2[::-1] not in stop_words) and not word2.isnumeric():
                x.append(bigram)
                y.append(count )
            if len(x) == n:
                break
        sns.barplot(x=y, y=x).set(xlabel='Frequency', ylabel='Bigram', title=title)

def plot_text_length_histogram(df):
    """
    Plot a histogram of the text length in the df.
    """
    words_lists = [text.split() for text in df]
    word_counts = [len(words_list) for words_list in words_lists]
    plt.hist(word_counts, bins=30)
    plt.xlabel('Text Length'), plt.ylabel('Frequency')
    plt.title("Question 2b - Text length histogram")
    plt.show()

def get_relevant_speakers(df: pd.DataFrame):
    relevant_speakers = []
    irrelevant_speakers = ['מזכירת הכנסת ירדנה מלר-הורביץ:', ' << דובר_המשך >>', 'קצרנית פרלמנטרית:',
                           'אתי בן-יוסף:', 'מנהלת הוועדה:']
    for speaker in list(df.Speaker.unique()):
        if (len(speaker) < 35 or (len(speaker) < 50 and (
                '<< דובר >>' in speaker or '<< יור >>' in speaker or '<< דובר_המשך >>' in speaker or 'שר' in speaker or 'שרת' in speaker or 'יו"ר' in speaker or 'בשם ועדת' in speaker))) \
                and 'אורח' not in speaker and 'קריאות' not in speaker and 'קריאה' not in speaker \
                and ' ' in speaker and speaker not in relevant_speakers and speaker not in irrelevant_speakers:
            relevant_speakers.append(speaker)
        elif speaker not in irrelevant_speakers:
            irrelevant_speakers.append(speaker)
    return relevant_speakers, irrelevant_speakers

def get_non_stop_words(df):
    stopwords = get_stopwords()
    return ' '.join([w for w in df.split() if w not in stopwords])


def get_lda_objects(non_stop_text):
    def _preprocess_text(non_stop_text):
        corpus = []
        for news in non_stop_text:
            words = news.split()
            corpus.append(words)
        return corpus

    corpus = _preprocess_text(non_stop_text)
    dic = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=6,
                                           id2word=dic,
                                           passes=10,
                                           workers=2)
    return lda_model, bow_corpus, dic


def show_topics(a, vocab, n_top_topics=10):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-n_top_topics-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]


def create_lda_model(dic, bow_corpus, topic_num, trial_num):
    # crate LDA model
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=topic_num, id2word=dic,
                                           passes=10, workers=2)
    return lda_model

def create_lda_topic_dict(lda_model, topic_num):
    topics_from_lda = lda_model.show_topics(num_topics=topic_num)
    topics_from_lda.sort(key=lambda x: x[0])
    lda_topic_dict = {topic[0]: topic[1:] for topic in topics_from_lda}
    for topic in lda_topic_dict:
        print(f'Topic {topic}: {lda_topic_dict[topic]}\n')
    return lda_topic_dict

def create_tag_df(lda_model, bow_corpus, corpus_orig, manual_topic_dict, lda_topic_dict, batch_num):
    tag_df = pd.DataFrame(columns=['NonStopwordsQuoteText', 'Topic', 'Topic_Prob', 'TopicWords'])
    for i in range(len(bow_corpus)):
        top_topics = lda_model.get_document_topics(bow_corpus[i])
        top_topics.sort(key=lambda x: x[1], reverse=True)
        topic, prob = top_topics[0]
        if topic in manual_topic_dict and prob > 0.5:
            tag_df.loc[i] = [corpus_orig[i], manual_topic_dict[topic], prob, lda_topic_dict[topic]]
    tag_df['Batch'] = batch_num
    tag_df.sort_values(by=['Topic_Prob'], ascending=False, inplace=True)
    tag_df.to_csv(f"../data/tagged_samples/tag_df_trial_{batch_num}.csv")
    return tag_df

def create_corpus(csv_path):
    samples = pd.read_csv(csv_path)
    corpus = [row.split() for row in samples['NonStopwordsQuoteText'].to_list()]
    corpus_orig = [row for row in samples['NonStopwordsQuoteText'].to_list()]
    dic = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    return corpus_orig, dic, bow_corpus