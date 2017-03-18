import re
from nltk.stem.porter import PorterStemmer

def preprocessing(comments):
    new_comments = []
    cache = {}
    stemmer = PorterStemmer()
    for comment in comments:
        comment = comment.lower()

        # sanitizing the data.
        comment = comment.replace("\\n", " ").replace("\\t", " ")
        comment = comment.replace("\\xa0", " ").replace("\\xc2", " ")
        # removing html tags
        tags_expr = re.compile('<.*?>')
        comment = re.sub(tags_expr, '', comment)

        # TODO - here we pruned all the urls. Perhaps, we should count the occurences of urls in a comment and use that as a feature.
        # removing urls
        url_expr = re.compile('http\S+')
        comment = re.sub(url_expr, '', comment)
        # expanding short forms
        comment = comment.replace(" u ", " you ").replace(" em ", " them ").replace(" da ", " the ").\
            replace(" yo ", " you ").replace(" ur ", " your ")
        comment = comment.replace("won't", "will not").replace("can't", "can not").replace("don't", "do not").\
            replace("i'm", "i am").replace("im", "i am").replace("ain't", "is not").replace("ll", "will").\
            replace("'t", " not").replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'d", " would")
        '''
        # stemming
        new_comment = []
        for word in comment.split(" "):
            if word not in cache:
                if word == "aed":
                    cache[word] = word
                else:
                    cache[word] = stemmer.stem(word)
            new_comment.append(cache[word])
        print new_comment
        new_comments.append(" ".join(new_comment))
        '''
        # Custom Stemming
        comment = re.sub("ies( |$)", "y ", comment)
        comment = re.sub("s( |$)", " ", comment)
        comment = re.sub("ing( |$)", " ", comment)
        comment = re.sub("ed( |$)", " ", comment)

        # Generalizing custom abuses.
        comment = re.sub(" [*$%&#@][*$%&#@]+", " xexp ", comment)
        comment = re.sub(" [0-9]+ ", " DD ", comment)

        # TODO - Mapping different forms of abuse to their root forms (like f00l, fo0l to Fool). Perhaps this wont be required due to use of char n grams as features.
        new_comments.append(comment)
    return new_comments



