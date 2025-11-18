def expand_contractions(text: str, contraction_map: dict):
    for contraction, expanded in contraction_map.items():
        text = text.replace(contraction, expanded)
    return text

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    # Expand contractions
    text = expand_contractions(text)
    
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercase and keep only alphabetic words
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
def chatbot_reply(user_query, unique_words, idf, tf_idf_matrix, unique_df, df):
    user_query = preprocess_text(user_query).split()
    tf = Counter(user_query)
    length = len(user_query)
    data,cols = [],[]
    for word,count in tf.items():
        if word in unique_words:
            data.append((count/length)*idf[word])
            cols.append(unique_words[word])
    query_vec = csr_matrix((data, ([0]*len(cols), cols)), shape=(1, len(unique_words)))
    similarity = cosine_similarity(query_vec, tf_idf_matrix).flatten()
    idx = similarity.argmax()
    print(idx)
    Id = unique_df.iloc[idx]['Id']
    best_ans = df[df['Id'] == Id].sort_values('Score_answer',ascending=False).iloc[0]
    return best_ans["Body_answer"], best_ans["question"], similarity[idx]

import numpy as np
def top_n_results(user_query, unique_words, idf, tf_idf_matrix, unique_df, df,n=1,score_req = 10):
    user_query = preprocess_text(user_query).split()
    tf = Counter(user_query)
    length = len(user_query)
    data,cols = [],[]
    for word,count in tf.items():
        if word in unique_words:
            data.append((count/length)*idf[word])
            cols.append(unique_words[word])
    query_vec = csr_matrix((data, ([0]*len(cols), cols)), shape=(1, len(unique_words)))
    similarity = cosine_similarity(query_vec, tf_idf_matrix).flatten()
    idx = np.argsort(-similarity)[:n]
    values = -np.sort(-similarity)[:n]
    idx_values = pd.DataFrame({'Id':unique_df.iloc[idx]['Id'], 'Similarity':values})
    Id = unique_df.iloc[idx]['Id']
    best_ans = df[df['Id'].isin(Id)]
    best_ans = best_ans[best_ans['Score_answer']>=score_req]
    return best_ans.merge(idx_values, on='Id')

import matplotlib.pyplot as plt
def Interval_Search(lower_bound, upper_bound, lowest_question_score, df):
    return df[(df["Score_ratio"] >= lower_bound) & (df["Score_ratio"] <= upper_bound) & (df["Score_question"] > lowest_question_score)].sort_values(by="Score_answer", ascending=False)

def bin_count(a,b, df):
    bin_counts = (df["Score_ratio"] >= a) & (df["Score_ratio"] < b)
    count = bin_counts.sum()
    return count

def plotting_searched_df(lower_bound, upper_bound, lowest_question_score):
    Searched_df = Interval_Search(lower_bound,upper_bound,lowest_question_score)
    plt.figure(figsize=(12,8))
    plt.scatter(Searched_df["Score_question"], Searched_df["Score_answer"], alpha=0.6)
    plt.xlabel("Question Score")
    plt.ylabel("Answer Score")
    plt.title("Question vs Answer Scores in Searched Data Frame")

    plt.figure(figsize=(12,8))
    plt.scatter(Searched_df["Score_ratio"], Searched_df["Score_question"], alpha=0.6)
    plt.xlabel("Ratio")
    plt.ylabel("Question Score")
    plt.title("Question vs Answer Scores in Searched Data Frame")

    return Searched_df