{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "pycharm": {
     "is_executing": false
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /home/jovyan/nltk_data...\n",
      "[nltk_data]   Unzipping tokenizers/punkt.zip.\n"
     ]
    }
   ],
   "source": [
    "import pathlib\n",
    "\n",
    "import numpy as np\n",
    "import scipy.sparse\n",
    "import scipy.io\n",
    "import pandas as pd\n",
    "import pickle\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "import networkx as nx\n",
    "import utils.preprocess\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stopwords\n",
    "from nltk import word_tokenize\n",
    "from nltk.corpus import stopwords as nltk_stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "import nltk\n",
    "nltk.download('punkt')\n",
    "from utils.data import load_glove_vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "save_prefix = 'data/preprocessed/DBLP_processed/'\n",
    "num_ntypes = 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading GloVe pretrained word vectors\n",
      "Done. 400000 words loaded!\n"
     ]
    }
   ],
   "source": [
    "author_label = pd.read_csv('data/raw/DBLP/author_label.txt', sep='\\t', header=None, names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')\n",
    "paper_author = pd.read_csv('data/raw/DBLP/paper_author.txt', sep='\\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')\n",
    "paper_conf = pd.read_csv('data/raw/DBLP/paper_conf.txt', sep='\\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')\n",
    "paper_term = pd.read_csv('data/raw/DBLP/paper_term.txt', sep='\\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')\n",
    "papers = pd.read_csv('data/raw/DBLP/paper.txt', sep='\\t', header=None, names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')\n",
    "terms = pd.read_csv('data/raw/DBLP/term.txt', sep='\\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')\n",
    "confs = pd.read_csv('data/raw/DBLP/conf.txt', sep='\\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')\n",
    "\n",
    "glove_dim = 50\n",
    "glove_vectors = load_glove_vectors(dim=glove_dim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# filter out all nodes which does not associated with labeled authors\n",
    "labeled_authors = author_label['author_id'].to_list()\n",
    "paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)\n",
    "valid_papers = paper_author['paper_id'].unique()\n",
    "papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)\n",
    "paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)\n",
    "paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)\n",
    "valid_terms = paper_term['term_id'].unique()\n",
    "terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# term lemmatization and grouping\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "lemma_id_mapping = {}\n",
    "lemma_list = []\n",
    "lemma_id_list = []\n",
    "i = 0\n",
    "for _, row in terms.iterrows():\n",
    "    i += 1\n",
    "    lemma = lemmatizer.lemmatize(row['term'])\n",
    "    lemma_list.append(lemma)\n",
    "    if lemma not in lemma_id_mapping:\n",
    "        lemma_id_mapping[lemma] = row['term_id']\n",
    "    lemma_id_list.append(lemma_id_mapping[lemma])\n",
    "terms['lemma'] = lemma_list\n",
    "terms['lemma_id'] = lemma_id_list\n",
    "\n",
    "term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}\n",
    "lemma_id_list = []\n",
    "for _, row in paper_term.iterrows():\n",
    "    lemma_id_list.append(term_lemma_mapping[row['term_id']])\n",
    "paper_term['lemma_id'] = lemma_id_list\n",
    "\n",
    "paper_term = paper_term[['paper_id', 'lemma_id']]\n",
    "paper_term.columns = ['paper_id', 'term_id']\n",
    "paper_term = paper_term.drop_duplicates()\n",
    "terms = terms[['lemma_id', 'lemma']]\n",
    "terms.columns = ['term_id', 'term']\n",
    "terms = terms.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# filter out stopwords from terms\n",
    "stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))\n",
    "stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()\n",
    "paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)\n",
    "terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# remove terms not found in GloVe\n",
    "#terms_not_in_glove = []\n",
    "#for _, row in terms.iterrows():\n",
    "#    if row['term'] not in glove_vectors:\n",
    "#        terms_not_in_glove.append(row['term'])\n",
    "#term_ids_not_in_glove = terms[terms['term'].isin(terms_not_in_glove)]['term_id'].to_list()\n",
    "#terms = terms[~(terms['term'].isin(terms_not_in_glove))].reset_index(drop=True)\n",
    "#paper_term = paper_term[~(paper_term['term_id'].isin(term_ids_not_in_glove))].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# consider only terms associated with at least two papers\n",
    "# if having meaningful word vectors for terms (e.g. GloVe)?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "author_label = author_label.sort_values('author_id').reset_index(drop=True)\n",
    "papers = papers.sort_values('paper_id').reset_index(drop=True)\n",
    "terms = terms.sort_values('term_id').reset_index(drop=True)\n",
    "confs = confs.sort_values('conf_id').reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# extract labels of authors\n",
    "labels = author_label['label'].to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# build the adjacency matrix for the graph consisting of authors, papers, terms and conferences\n",
    "# 0 for authors, 1 for papers, 2 for terms, 3 for conferences\n",
    "dim = len(author_label) + len(papers) + len(terms) + len(confs)\n",
    "type_mask = np.zeros((dim), dtype=int)\n",
    "type_mask[len(author_label):len(author_label)+len(papers)] = 1\n",
    "type_mask[len(author_label)+len(papers):len(author_label)+len(papers)+len(terms)] = 2\n",
    "type_mask[len(author_label)+len(papers)+len(terms):] = 3\n",
    "\n",
    "author_id_mapping = {row['author_id']: i for i, row in author_label.iterrows()}\n",
    "paper_id_mapping = {row['paper_id']: i + len(author_label) for i, row in papers.iterrows()}\n",
    "term_id_mapping = {row['term_id']: i + len(author_label) + len(papers) for i, row in terms.iterrows()}\n",
    "conf_id_mapping = {row['conf_id']: i + len(author_label) + len(papers) + len(terms) for i, row in confs.iterrows()}\n",
    "\n",
    "adjM = np.zeros((dim, dim), dtype=int)\n",
    "for _, row in paper_author.iterrows():\n",
    "    idx1 = paper_id_mapping[row['paper_id']]\n",
    "    idx2 = author_id_mapping[row['author_id']]\n",
    "    adjM[idx1, idx2] = 1\n",
    "    adjM[idx2, idx1] = 1\n",
    "for _, row in paper_term.iterrows():\n",
    "    idx1 = paper_id_mapping[row['paper_id']]\n",
    "    idx2 = term_id_mapping[row['term_id']]\n",
    "    adjM[idx1, idx2] = 1\n",
    "    adjM[idx2, idx1] = 1\n",
    "for _, row in paper_conf.iterrows():\n",
    "    idx1 = paper_id_mapping[row['paper_id']]\n",
    "    idx2 = conf_id_mapping[row['conf_id']]\n",
    "    adjM[idx1, idx2] = 1\n",
    "    adjM[idx2, idx1] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)\n",
    "# mat = scipy.io.loadmat('data/raw/DBLP/DBLP4057_GAT_with_idx.mat') # R \n",
    "mat = scipy.io.loadmat('data/HAN_mat_files/DBLP4057_GAT_with_idx.mat')\n",
    "features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])\n",
    "features_author = scipy.sparse.csr_matrix(features_author)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['__header__', '__version__', '__globals__', 'net_APTPA', 'label', 'train_idx', 'features', 'net_APCPA', 'net_APA', 'test_idx', 'val_idx'])\n"
     ]
    }
   ],
   "source": [
    "# R : Added to analyze the content of matlab file provided by HAN authors (sent by email from MAGNN authors)   \n",
    "import scipy.io as sio\n",
    "data = sio.loadmat('data/HAN_mat_files/DBLP4057_GAT_with_idx.mat')\n",
    "print(data.keys())\n",
    "# End R "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jovyan/envs/magnn_env/lib/python3.7/site-packages/sklearn/feature_extraction/text.py:300: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [\"'d\", \"'ll\", \"'m\", \"'re\", \"'s\", \"'ve\", 'doe', 'ha', 'le', \"n't\", 'need', 'sha', 'u', 'wa', 'wo'] not in stop_words.\n",
      "  'stop_words.' % sorted(inconsistent))\n"
     ]
    }
   ],
   "source": [
    "# use bag-of-words representation of paper titles as the features of papers\n",
    "class LemmaTokenizer:\n",
    "    def __init__(self):\n",
    "        self.wnl = WordNetLemmatizer()\n",
    "    def __call__(self, doc):\n",
    "        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]\n",
    "vectorizer = CountVectorizer(min_df=2, stop_words=stopwords, tokenizer=LemmaTokenizer())\n",
    "features_paper = vectorizer.fit_transform(papers['paper_title'].values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# use pretrained GloVe vectors as the features of terms\n",
    "features_term = np.zeros((len(terms), glove_dim))\n",
    "for i, row in terms.iterrows():\n",
    "    features_term[i] = glove_vectors.get(row['term'], glove_vectors['the'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": true,
     "name": "#%%\n"
    }
   },
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_13820/1547456699.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;31m# get metapath based neighbor pairs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m     \u001b[0mneighbor_pairs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpreprocess\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_metapath_neighbor_pairs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0madjM\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtype_mask\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexpected_metapaths\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m     \u001b[0;31m# construct and save metapath-based networks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m     \u001b[0mG_list\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpreprocess\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_networkx_graph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mneighbor_pairs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtype_mask\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/MAGNN-master/utils/preprocess.py\u001b[0m in \u001b[0;36mget_metapath_neighbor_pairs\u001b[0;34m(M, type_mask, expected_metapaths)\u001b[0m\n\u001b[1;32m     45\u001b[0m                 \u001b[0mhas_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m                 single_source_paths = nx.single_source_shortest_path(\n\u001b[0;32m---> 47\u001b[0;31m                     partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)\n\u001b[0m\u001b[1;32m     48\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mtarget\u001b[0m \u001b[0;32min\u001b[0m \u001b[0msingle_source_paths\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     49\u001b[0m                     \u001b[0mhas_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/envs/magnn_env/lib/python3.7/site-packages/networkx/algorithms/shortest_paths/unweighted.py\u001b[0m in \u001b[0;36msingle_source_shortest_path\u001b[0;34m(G, source, cutoff)\u001b[0m\n\u001b[1;32m    341\u001b[0m     \u001b[0mnextlevel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0msource\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m}\u001b[0m     \u001b[0;31m# list of nodes to check at next level\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    342\u001b[0m     \u001b[0mpaths\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0msource\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0msource\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m}\u001b[0m  \u001b[0;31m# paths dictionary  (paths to key from source)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 343\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mdict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_single_shortest_path\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mG\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnextlevel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpaths\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcutoff\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mjoin\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    344\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    345\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/envs/magnn_env/lib/python3.7/site-packages/networkx/algorithms/shortest_paths/unweighted.py\u001b[0m in \u001b[0;36m_single_shortest_path\u001b[0;34m(adj, firstlevel, paths, cutoff, join)\u001b[0m\n\u001b[1;32m    369\u001b[0m         \u001b[0mnextlevel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    370\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mv\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mthislevel\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 371\u001b[0;31m             \u001b[0;32mfor\u001b[0m \u001b[0mw\u001b[0m \u001b[0;32min\u001b[0m \u001b[0madj\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mv\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    372\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mw\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mpaths\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    373\u001b[0m                     \u001b[0mpaths\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpaths\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mv\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/envs/magnn_env/lib/python3.7/site-packages/networkx/classes/coreviews.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, name)\u001b[0m\n\u001b[1;32m     79\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__getitem__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 81\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mAtlasView\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_atlas\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     82\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/envs/magnn_env/lib/python3.7/site-packages/networkx/classes/coreviews.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, d)\u001b[0m\n\u001b[1;32m     43\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     44\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__init__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0md\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 45\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_atlas\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     46\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     47\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "expected_metapaths = [\n",
    "    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],\n",
    "    [(1, 0, 1), (1, 2, 1), (1, 3, 1)],\n",
    "    [(2, 1, 2), (2, 1, 0, 1, 2), (2, 1, 3, 1, 2)],\n",
    "    [(3, 1, 3), (3, 1, 0, 1, 3), (3, 1, 2, 1, 3)]\n",
    "]\n",
    "# create the directories if they do not exist\n",
    "for i in range(1):\n",
    "    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)\n",
    "for i in range(1):\n",
    "    # get metapath based neighbor pairs\n",
    "    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])\n",
    "    # construct and save metapath-based networks\n",
    "    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)\n",
    "    \n",
    "    # save data\n",
    "    # networkx graph (metapath specific)\n",
    "    for G, metapath in zip(G_list, expected_metapaths[i]):\n",
    "        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')\n",
    "    # node indices of edge metapaths\n",
    "    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)\n",
    "    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):\n",
    "        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)\n",
    "# save data\n",
    "# all nodes adjacency matrix\n",
    "scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))\n",
    "# all nodes (authors, papers, terms and conferences) features\n",
    "# currently only have features of authors, papers and terms\n",
    "scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)\n",
    "scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)\n",
    "np.save(save_prefix + 'features_{}.npy'.format(2), features_term)\n",
    "# all nodes (authors, papers, terms and conferences) type labels\n",
    "np.save(save_prefix + 'node_types.npy', type_mask)\n",
    "# author labels\n",
    "np.save(save_prefix + 'labels.npy', labels)\n",
    "# author train/validation/test splits\n",
    "rand_seed = 1566911444\n",
    "train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)\n",
    "train_idx, test_idx = train_test_split(train_idx, test_size=3257, random_state=rand_seed)\n",
    "train_idx.sort()\n",
    "val_idx.sort()\n",
    "test_idx.sort()\n",
    "np.savez(save_prefix + 'train_val_test_idx.npz',\n",
    "         val_idx=val_idx,\n",
    "         train_idx=train_idx,\n",
    "         test_idx=test_idx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-02T08:48:33.205617Z",
     "iopub.status.busy": "2026-06-02T08:48:33.202745Z",
     "iopub.status.idle": "2026-06-02T08:48:35.096979Z",
     "shell.execute_reply": "2026-06-02T08:48:35.092069Z",
     "shell.execute_reply.started": "2026-06-02T08:48:33.205547Z"
    }
   },
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'gcd' from 'fractions' (/usr/lib/python3.10/fractions.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Added for statistics \u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[0;32m----> 3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mutils\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdata\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_DBLP_data   \u001b[38;5;66;03m# or load_DBLP_data\u001b[39;00m\n\u001b[1;32m      5\u001b[0m nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx \u001b[38;5;241m=\u001b[39m load_IMDB_data()\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# number of nodes\u001b[39;00m\n",
      "File \u001b[0;32m/media/rihab/TOSHIBA EXT/2026/ExperimentsECE/Aggregation_Codes/Used_codes/MAGNN-master/utils/data.py:1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnx\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mscipy\u001b[39;00m\n",
      "File \u001b[0;32m~/.local/lib/python3.10/site-packages/networkx/__init__.py:114\u001b[0m\n\u001b[1;32m    111\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mrelabel\u001b[39;00m\n\u001b[1;32m    112\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mrelabel\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m--> 114\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\n\u001b[1;32m    115\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m    117\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mreadwrite\u001b[39;00m\n",
      "File \u001b[0;32m~/.local/lib/python3.10/site-packages/networkx/generators/__init__.py:14\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mexpanders\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     13\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgeometric\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m---> 14\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mintersection\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     15\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mjoint_degree_seq\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mgenerators\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlattice\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n",
      "File \u001b[0;32m~/.local/lib/python3.10/site-packages/networkx/generators/intersection.py:13\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mrandom\u001b[39;00m\n\u001b[1;32m     12\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnx\u001b[39;00m\n\u001b[0;32m---> 13\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m bipartite\n\u001b[1;32m     14\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mutils\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m py_random_state\n\u001b[1;32m     16\u001b[0m __author__ \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mjoin([\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mAric Hagberg (hagberg@lanl.gov)\u001b[39m\u001b[38;5;124m'\u001b[39m])\n",
      "File \u001b[0;32m~/.local/lib/python3.10/site-packages/networkx/algorithms/__init__.py:16\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcycles\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     15\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcuts\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[0;32m---> 16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdag\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     17\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdistance_measures\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n\u001b[1;32m     18\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnetworkx\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malgorithms\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdistance_regular\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;241m*\u001b[39m\n",
      "File \u001b[0;32m~/.local/lib/python3.10/site-packages/networkx/algorithms/dag.py:23\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[38;5;124;03m\"\"\"Algorithms for directed acyclic graphs (DAGs).\u001b[39;00m\n\u001b[1;32m     16\u001b[0m \n\u001b[1;32m     17\u001b[0m \u001b[38;5;124;03mNote that most of these functions are only guaranteed to work for DAGs.\u001b[39;00m\n\u001b[1;32m     18\u001b[0m \u001b[38;5;124;03mIn general, these functions do not check for acyclic-ness, so it is up\u001b[39;00m\n\u001b[1;32m     19\u001b[0m \u001b[38;5;124;03mto the user to check for that.\u001b[39;00m\n\u001b[1;32m     20\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     22\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mcollections\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m defaultdict, deque\n\u001b[0;32m---> 23\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfractions\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m gcd\n\u001b[1;32m     24\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfunctools\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m partial\n\u001b[1;32m     25\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mitertools\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m chain\n",
      "\u001b[0;31mImportError\u001b[0m: cannot import name 'gcd' from 'fractions' (/usr/lib/python3.10/fractions.py)"
     ]
    }
   ],
   "source": [
    "# Added for statistics \n",
    "import numpy as np\n",
    "from utils.data import load_DBLP_data   # or load_DBLP_data\n",
    "\n",
    "nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()\n",
    "\n",
    "# number of nodes\n",
    "num_nodes = len(type_mask)\n",
    "\n",
    "# number of edges\n",
    "num_edges = adjM.nnz   # number of non-zero entries in adjacency matrix\n",
    "\n",
    "# number of classes\n",
    "num_classes = len(np.unique(labels))\n",
    "\n",
    "print(\"Nodes:\", num_nodes)\n",
    "print(\"Edges:\", num_edges)\n",
    "print(\"Classes:\", num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "# post-processing for mini-batched training\n",
    "target_idx_list = np.arange(4057)\n",
    "for metapath in [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]:\n",
    "    edge_metapath_idx_array = np.load(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.npy')\n",
    "    target_metapaths_mapping = {}\n",
    "    for target_idx in target_idx_list:\n",
    "        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[edge_metapath_idx_array[:, 0] == target_idx][:, ::-1]\n",
    "    out_file = open(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb')\n",
    "    pickle.dump(target_metapaths_mapping, out_file)\n",
    "    out_file.close()\n",
    "print(\"end of preprocessing\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "pycharm": {
   "stem_cell": {
    "cell_type": "raw",
    "metadata": {
     "collapsed": false
    },
    "source": []
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
