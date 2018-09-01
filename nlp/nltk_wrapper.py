#!/usr/bin/python
# encoding: utf-8
import nltk
from nltk.util import trigrams
from nltk.util import bigrams
# nltk中有以下几个概念
# raw：原始文本，可以看作一个大的string
# text
# words：word list。[w1,w2,w3...]
# sents：句子list。以'. '和换行符'\n'等作为分隔符的词链表。如下形式：
#     [[w1,w2...],[w3,w4,...],...]
# 相互转换
# raw转text: 需要先转成words list, 如 nltk.Text(raw.split())
# raw转words list：raw.split()
# raw转sents list: ?
# text转raw：''.join(list(text))
# text转words list: list(text)
# text转sents：？
# words转raw：''.join(words)
# words转text: nltk.Text(words)
# words转sents: ?
# sents转words: [w for s in sents for w in s]
# sents转raw
# sents转text


class nltk_wrapper(object):
    def __init__(self, text):
        '''
        text: 一个nltk.text.Text类型的语料
        '''
        self._text = text
    def english_stopwords(self):
        '''
        @return: english停用词list
        '''
        return nltk.corpus.stopwords.words('english')
    #文本探索
    #输入一个nltk.text.Text类型的语料，进行一些统计分析
    def length(self):
        '''
        返回文本的总长度（不包含空格）
        '''
        return len(self._text)
    def vocabulary(self):
        '''
        按照字母升序返回文本的词汇列表
        '''
        return sorted(set(self._text))
    def countword(self, word):
        '''
        返回某个word在text中出现的次数
        '''
        return self._text.count(word)
    def words_freq(self, n):
        '''
        返回text的词频字典
        List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.
        等价于nltk.book.FreqDist(self._text).most_common()
        '''        
        return self._text.vocab().most_common(n)
    
    def lexical_diversity(self): 
        '''
        返回重复词密度：平均每个词重复几次
        '''        
        return len(self._text) / len(set(self._text)) 
    
    def percentage(self, word): 
        '''
        返回关键词密度
        '''        
        return self.countword(word)/self.length
    
    #文本处理
    def get_bigrams(self):
        return bigrams(self._text)
    def get_trigrams(self):
        return trigrams(self._text)
    def get_words(self, length):
        '''
        返回text中所有length长度的word
        '''        
        return [word for word in self._text if len(word)==length]
    #载入自己的语料库
    def load_corpus(self, corpus_root, regex='.*', encoding='utf8'):
        '''
        返回自己的语料库
        corpus_root：语料所在的目录
        regex：正则，默认导入目录下所有文件
        '''        
        from nltk.corpus import PlaintextCorpusReader
        wordlists = PlaintextCorpusReader(corpus_root, regex, encoding= encoding)
        return wordlists
    '''
    ########################################
    POS词性标注
    NLTK提供了4种标注器
    * nltk.DefaultTagger(tag)
    A tagger that assigns the same tag to every token.
    * 
    ########################################
    '''
    def word_tokenize(self, text, language='english', preserve_line=False):
        '''
        英文分词
        @text: str，字符串文本
        @return: list
        '''
        return nltk.word_tokenize(text, language, preserve_line)
    def pos_tag(self, tokens, tagset=None, lang='eng'):
        '''
        词性标注器
        Use NLTK's currently recommended part of speech tagger to
        tag the given list of tokens.
        >>> from nltk.tag import pos_tag
        >>> from nltk.tokenize import word_tokenize
        >>> pos_tag(word_tokenize("John's big idea isn't all that bad."))
        [('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
        ("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]
        '''
        return nltk.pos_tag(tokens, tagset, lang)
    
    def upenn_tagset(self, tagpattern=None):
        '''
        解释标注tag的意思
        nltk.help.upenn_tagset('RB')
        RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        '''
        return nltk.help.upenn_tagset(tagpattern)
    @classmethod
    def get_brown_tagged_words(self, fileids=None, categories=None, tagset=None):
        '''
        nltk中自带的标注语料库
        @tagset: #如果要使用通用的简化的标注，可以使用参数tagset='universal'
        @return: [(w1,tag1)...] 
        '''
        return nltk.corpus.brown.tagged_words(fileids, categories, tagset)
    @classmethod
    def get_brown_tagged_sents(self, fileids=None, categories=None, tagset=None):
        '''
        nltk中自带的标注语料库
        @tagset: #如果要使用通用的简化的标注，可以使用参数tagset='universal'
        @return: [[(w1,tag1)...],[(wn,tagn),...]...] 
        '''
        return nltk.corpus.brown.tagged_sents(fileids, categories, tagset)
    def DefaultTagger(self, tag):
        '''
        A tagger that assigns the same tag to every token.
        >>> from nltk.tag import DefaultTagger
        >>> default_tagger = DefaultTagger('NN')
        >>> default_tagger.tag('This is a test'.split())
        [('This', 'NN'), ('is', 'NN'), ('a', 'NN'), ('test', 'NN')]
        '''
        return nltk.DefaultTagger(tag)
    
    def RegexpTagger(self, regexps, backoff=None):
        '''
        Regular Expression Tagger

        The RegexpTagger assigns tags to tokens by comparing their
        word strings to a series of regular expressions.  
        #正则表达式标注器
        patterns = [
            (r'.*ing$', 'VBG'), # gerunds
            (r'.*ed$', 'VBD'), # simple past
            (r'.*es$', 'VBZ'), # 3rd singular present
            (r'.*ould$', 'MD'), # modals
            (r'.*\'s$', 'NN$'), # possessive nouns
            (r'.*s$', 'NNS'), # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
            (r'.*', 'NN') # nouns (default)
        ]
        regexp_tagger = nltk.RegexpTagger(patterns)
        regexp_tagger.tag(text)
        '''
        return nltk.RegexpTagger(regexps, backoff)
    
    def UnigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False):
        '''
        Unigram Tagger 选出train中word最可能的词性作为该word的词性
        
        The UnigramTagger finds the most likely tag for each word in a training
        corpus, and then uses that information to assign tags to new tokens.
        @:param train: The corpus of training data, a list of tagged sentences
        :type train: list(list(tuple(str, str)))
        :param model: The tagger model
        :type model: dict
        :param backoff: Another tagger which this tagger will consult when it is
            unable to tag a word
        :type backoff: TaggerI
        :param cutoff: The number of instances of training data the tagger must see
            in order not to use the backoff tagger
        :type cutoff: int
            >>> from nltk.corpus import brown
            >>> from nltk.tag import UnigramTagger
            >>> test_sent = brown.sents(categories='news')[0]
            >>> unigram_tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
            >>> for tok, tag in unigram_tagger.tag(test_sent):
            ...     print("(%s, %s), " % (tok, tag))
        '''
        return nltk.UnigramTagger(train, model, backoff, cutoff, verbose)
    
    def nltk.BigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False):
        '''
        考虑前一个word的tag来确定自身的tag
        A tagger that chooses a token's tag based its word string and on
        the preceding words' tag.  In particular, a tuple consisting
        of the previous tag and the word is looked up in a table, and
        the corresponding tag is returned.

        '''
        return nltk.BigramTagger(train=None, model=None, backoff=None, cutoff=0, verbose=False)
    
    
    
    
    
    
    
    
    
    