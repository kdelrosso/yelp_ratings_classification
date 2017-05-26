## Getting the Data

- The original [Yelp data](https://www.yelp.com/dataset_challenge)
- Sentimate analysis dataset #1, [SentiWordNet](http://sentiwordnet.isti.cnr.it)
- Sentimate analysis dataset #2, [Hu and Liu's Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)

To convert the original json to csv we used a script available [here](https://github.com/Yelp/dataset-examples) and ran

```
$ python json_to_csv_converter.py ../data/yelp_academic_dataset_review.json
```

We created a PostgresSQL database to store the data

```
$ psql -c "CREATE DATABASE yelp;"
```

and created a `reviews` table

```
DROP TABLE IF EXISTS reviews;

CREATE TABLE reviews (
    user_id             varchar(100),
    review_id           varchar(100),
    text                text,
    votes_cool          int,
    business_id         varchar(100),
    votes_funny         int,
    stars               int,
    date                timestamp,
    type                varchar(100),
    votes_useful        int,
    primary key(review_id)
);

COPY reviews FROM '../data/yelp_academic_dataset_review.csv' CSV HEADER;
```

copying data from the .csv file created above.

## Running the Code

All code can be run from `main.py`. We can load and preprocess the Yelp reviews data using

```
df = get_reviews_data(PROJECT_DIR + '/data/saved_df.pkl', use_saved=False)
```

which calls functions in `utils/preprocessing_utils.py`. We can create exploratory data analysis plots with

```
show_all_eda(df)
```

which calls functions in `utils/eda_utils.py`. We can split our data into training, test, and validation sets and use grid search to optimize stage 1 & 2 model parameters using

```
X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df[KEEP_COLS], 'stars')
grid_search_naive_bayes_with_tfidf(df_train['text'].values, df_train['stars'].values)
grid_search_stage_two(X_train, y_train)
```

which calls functions in `utils/model_utils.py`. We can fit and print classification results for the two stage TF-IDF model using

```
two_stage_model(X_train, y_train, X_test, y_test)
```

which calls the `NaiveBayesTwoStageClassifier` class in `two_stage_tfidf_model.py`. Finally we can fit and use the Doc2Vec model with

```
yelp_model = doc2vec_model(df, train_dict, test_dict, k=25)
doc2vec_examples(yelp_model)
```

which calls the `Doc2VecClassifier` class in `doc2vec_model.py`.
