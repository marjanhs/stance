import html, json
import subprocess
from idna import intranges
from statsmodels.genmod.families.links import probit
import subprocess
from fastai.text import *
import collections
re1 = re.compile(r'  +')
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
SOH = '\x01'
from nltk import FreqDist
from collections import defaultdict
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('\x01','')
    return re1.sub(' ', html.unescape(x))

def fixup_two_context(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    x= re1.sub(' ', html.unescape(x))
    x= x.split('\x01')
    if len(x) ==2:
        return [x[0],x[1]]
    elif len(x) in {3,4}:
        return [x[1], x[2]]
    else:
        raise Exception("No Valid Input", len(x), x[-1])


def get_texts(df, col_no=1, is_twocontext=False, sent_tok=False, type=False):
    labels = df.iloc[:, 1].values.astype(np.int64)  # id, label, context, question, opinion
    paths = df.iloc[:, 0].values

    if type is None:
        issues = [paths[i].split('\\')[5] for i in range(len(paths))]
        ids = [paths[i].split('/')[1].replace('.txt', '.' + issues[i]) for i in range(len(paths))]
    elif type=='ibm':
        issues = [paths[i].split('_')[1] for i in range(len(paths))]
        ids = [paths[i].split('_')[0].replace('.txt', '.' + issues[i]) for i in range(len(paths))]
    elif type=='twitter':
        issues =[df.iloc[:,2][i] for i in range(len(df.iloc[:, 2]))]
        ids = [df.iloc[:, 0][i] for i in range(len(df.iloc[:, 0]))]



    texts =  df[col_no].astype(str)
    #texts = f'\n{BOS} {FLD} 1 ' + df[col_no].astype(str)
    # texts += f' {FLD} {i-n_lbls} ' + df[col_no].astype(str)
    # for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    if is_twocontext and col_no==2:
        texts = list(texts.apply(fixup_two_context).values)
        texts_pro = [e[0] for e in texts]
        texts_con = [e[1] for e in texts]
        tok_pro = Tokenizer().proc_all_mp(partition_by_cores(texts_pro))
        tok_con = Tokenizer().proc_all_mp(partition_by_cores(texts_con))
        return (tok_pro, tok_con), list(labels), issues, ids
    else:
        texts = list(texts.apply(fixup).values)
        tok = SentTokenizer().proc_all_mp(partition_by_cores(texts)) if sent_tok\
            else Tokenizer().proc_all_mp(partition_by_cores(texts))
        #tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
        #tok = SentTokenizer().proc_all_mp(partition_by_cores(texts))
        return tok, list(labels), issues,  ids


class SentTokenizer():

    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self,x): return self.re_br.sub("\n", x)

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP,TOK_SENT,TOK_MIX = ' t_up ',' t_st ',' t_mx '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
    #                 else [TOK_SENT,s.lower()] if (s.istitle() and re_word.search(prev))
                    else [s.lower()])
    #         if re_nonsp.search(s): prev = s
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(SentTokenizer.replace_rep, s)
        s = self.re_word_rep.sub(SentTokenizer.replace_wrep, s)
        s = SentTokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = SentTokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en', ncpus = None):
        ncpus = ncpus or num_cpus()//2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(SentTokenizer.proc_all, ss, [lang]*len(ss)), [])

    def spacy_tok(self, x):
        return [[t.text for t in s] for s in self.tok(self.sub_br(x)).sents]
        #return [t.text for t in self.tok.tokenizer(self.sub_br(x))]


def get_all(df, col_list, is_twocontext=False, sent_tok=False, type=None):
    tokens = [[] for _ in col_list]
    tok, labels, issues, ids = [], [], [], []
    for j, r in enumerate(df):
        print(j)
        for i in range(len(col_list)):
            tok_, labels_, issues_, id_= get_texts(r, col_list[i], is_twocontext, sent_tok, type)
            tokens[i] += tok_;
        labels += labels_
        issues += issues_
        ids +=id_
    return tokens, labels, issues, ids


def to_balanced_train_val_tst(LM_PATH, file_name):
    args = pd.read_csv(LM_PATH / file_name, skiprows=1)
    rows = np.arange(0, len(args.iloc[:]))
    np.random.shuffle(rows)
    args = args.iloc[rows]
    pos = [i for i in range(len(args)) if args.iloc[i][1] == 1]
    neg = [i for i in range(len(args)) if args.iloc[i][1] == 0]

    np.random.shuffle(pos)
    np.random.shuffle(neg)
    m = min(len(pos), len(neg))
    pos = pos[0:m]
    neg = neg[0:m]
    assert len(pos) == len(neg)
    total = len(pos) + len(neg)
    print(f'# of +ve:{len(pos)}, # of -ve:{len(neg)}')
    train_len, val_len = int(int(.7 * total) / 2), int(int(.1 * total) /2)
    tst_len = int((total - (2 * train_len + 2 * val_len)) / 2)

    train_data = pd.concat([args.iloc[pos[0: train_len]], args.iloc[neg[0: train_len]]])
    val_data = pd.concat([args.iloc[pos[train_len: train_len + val_len]],
                            args.iloc[neg[train_len: train_len + val_len]]])
    assert len(val_data) == (len(pos[train_len: train_len + val_len]) + len(neg[train_len: train_len + val_len]))
    print(len(pos[train_len: train_len + val_len]), len(neg[train_len: train_len + val_len]))
    tst_data = pd.concat([args.iloc[pos[train_len + val_len:]], args.iloc[neg[train_len + val_len:]]])

    print('train: ', len(train_data), 'val: ', len(val_data), 'test:', len(tst_data))
    print('train:', get_no_labels(train_data), 'val:', get_no_labels(val_data),'test:', get_no_labels(tst_data))
    train_data.to_csv(LM_PATH / (file_name.replace('.csv', '_train.csv')), header=False, index=False)
    val_data.to_csv(LM_PATH / (file_name.replace('.csv', '_val.csv')), header=False, index=False)
    tst_data.to_csv(LM_PATH / (file_name.replace('.csv', '_tst.csv')), header=False, index=False)



def get_no_labels(dframe):
    p, n = 0, 0
    for i in range(len(dframe)):
        if dframe.iloc[i][1]==1:
            p+=1
        elif dframe.iloc[i][1]==0:
            n+=1
        else:
            raise ValueError('label value is not valid!')
    return p, n


def save_tokens(LM_PATH, pre_file_name, type=None, features=None, sentence_tok=False, sets=['train', 'val', 'tst']):
    '''
    convert documents to word tokens.
    :param LM_PATH: path to .csv dataset files. (train, val, tst)
    :param pre_file_name:  ds prefix file name without set name e.g. arg_quot for arg_quot_train.csv
    :param type: op or None. if op only opinion will be used for itos and doc representation
    :param features: dic of additional features to be created.
     Can include sentiment and file path of selected sentiment features.
    :param sentence_tok: indicates type of tokenizer which will be used for tokenizing the context and opinion
    docs (sentence tokenizer/ word tokenizer). If True each doc will be list of sentences and each sentence
     will be list of word tokens. If False each doc will be a list of word tokens
    :return:
    '''

    CL_PATH = Path(LM_PATH / (pre_file_name + '_clas'))
    #if type is not None : CL_PATH = Path(LM_PATH / (pre_file_name + '_clas') / type)
    CL_PATH.mkdir(exist_ok=True)
    if features is None:
        features = dict()
    features.update({'context': 0, 'question': 0, 'opinion' : 0, 'labels':0, 'issues':0, 'ids':0})
    chunksize = 24000
    tokens = dict()
    for s in sets:
        df = pd.read_csv(LM_PATH / (pre_file_name + f'_{s}.csv'), header=None, chunksize=chunksize)
        tok, labels, iss, ids = get_all(df, [2, 3,4], sent_tok=sentence_tok, type=type)
        tokens[s] = dict()
        tokens[s]['context'], tokens[s]['question'], tokens[s]['opinion'] = tok[0], tok[1], tok[2]
        tokens[s]['labels'], tokens[s]['issues'], tokens[s]['ids'] = labels, iss, ids
        print(s, [(e, len(tokens[s][e])) for e in tokens[s]])

    if type == 'op':
        tok_trn = [tokens['train']['opinion']]
        assert len(features) == 4
    else:
        tok_trn = [tokens['train']['opinion'] + tokens['train']['context']]

    # saving itos and stoi
    if sentence_tok:
        tok_trn = [s for d in tok_trn for s in d]
    tok_trn_all = [t for s in tok_trn for t in s]
    freq = Counter(p for o in tok_trn_all for p in o)
    max_vocab = 60000
    min_freq = 0

    itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    pickle.dump(itos, open(CL_PATH / 'itos.pkl', 'wb'))
    print(f'size of itos: {len(itos)}', itos[:10])
    for s in sets:
        for feature in features:
            if feature == 'sentiment':
                p = LM_PATH / f'{pre_file_name}_train_{features["sentiment"]}'
                top_sent_tokens = load_obj(p)
                tokens[s][feature] = [top_sent_tokens[tokens[s]['issues'][i]] for i in range(len(tokens[s]['issues']))]
                np.save(CL_PATH / f'tok_{s}_{feature}.npy', tokens[s][feature])
            elif feature in {"labels", "ids","issues"}:
                np.save(CL_PATH / f'ids_{s}_{feature}.npy',  tokens[s][feature])
            else:
                np.save(CL_PATH / f'tok_{s}_{feature}.npy', tokens[s][feature])
        print(s, [(e, len(tokens[s][e])) for e in tokens[s]])


def save_tokens_twocontext(LM_PATH, pre_file_name, type=None, features= None):
    sets = ['train', 'val', 'tst']
    CL_PATH = Path(LM_PATH / (pre_file_name + '_clas'))
    if type is not None : CL_PATH = Path(LM_PATH / (pre_file_name + '_clas') / type)
    CL_PATH.mkdir(exist_ok=True)
    if features is None:
        features = dict()
    features.update({'context_pro' : 0, 'context_con':0, 'question': 0, 'opinion' : 0, 'labels':0, 'issues':0, 'ids':0})
    chunksize = 24000
    tokens = dict()
    for s in sets:
        df = pd.read_csv(LM_PATH / (pre_file_name + f'_{s}.csv'), header=None, chunksize=chunksize)
        tok, labels, iss, ids = get_all(df, [2, 3,4], is_twocontext=True)
        tokens[s] = dict()
        tokens[s]['question'], tokens[s]['opinion'] = tok[1], tok[2]
        tokens[s]['context_pro'],  tokens[s]['context_con'] = tok[0]
        tokens[s]['labels'], tokens[s]['issues'], tokens[s]['ids'] = labels, iss, ids
        print(s, [(e, len(tokens[s][e])) for e in tokens[s]])

    if type == 'op':
        tok_trn = [tokens['train']['opinion']]
        assert len(features) == 4
    else:
        tok_trn = [tokens['train']['opinion'] + tokens['train']['context_pro'] + tokens['train']['context_con']]

    # saving itos and stoi
    tok_trn_all = [t for l in tok_trn for t in l]
    freq = Counter(p for o in tok_trn_all for p in o)
    max_vocab = 60000
    min_freq = 0

    itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    pickle.dump(itos, open(CL_PATH / 'itos.pkl', 'wb'))
    print(f'size of itos: {len(itos)}')
    for s in sets:
        for feature in features:
            if feature == 'sentiment':
                p = LM_PATH / f'{pre_file_name}_train_{features["sentiment"]}'
                top_sent_tokens = load_obj(p)
                tokens[s][feature] = [top_sent_tokens[tokens[s]['issues'][i]] for i in range(len(tokens[s]['issues']))]
                np.save(CL_PATH / f'tok_{s}_{feature}.npy', tokens[s][feature])
            elif feature in {"labels", "ids","issues"}:
                np.save(CL_PATH / f'ids_{s}_{feature}.npy',  tokens[s][feature])
            else:
                np.save(CL_PATH / f'tok_{s}_{feature}.npy', tokens[s][feature])
        print(s, [(e, len(tokens[s][e])) for e in tokens[s]])




def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_ids(LM_PATH, pre_file_name='', type=None, features=None, is_twocontext=False, sentence_tok=False, sets = ['train', 'val', 'tst']):
    CL_PATH = Path(LM_PATH / (pre_file_name + '_clas'))
    if type is not None :  CL_PATH = Path(LM_PATH / (pre_file_name + '_clas') / type)
    itos = pickle.load((CL_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    to_ids(CL_PATH=CL_PATH, type=type, stoi=stoi,features=features,
           is_twocontext=is_twocontext, sentence_tok=sentence_tok, sets=sets)


def to_ids(CL_PATH, type=None, stoi=None, features=None, is_twocontext=False, sentence_tok=False, sets=['train', 'val', 'tst'] ):
    '''
    convert tokens to ids (indices).
    :param CL_PATH:
    :param type:
    :param stoi:
    :param features:
    :param is_twocontext:
    :param sentence_tok: indicates type of tokenizer which was used for tokenizing the context and opinion
    docs (sentence tokenizer/ word tokenizer). If True each doc is list of sentences and each sentence
     is list of word tokens. If False each doc is a list of word tokens
    :return:
    '''


    if features is None:
        features = dict()
    features['opinion']=0
    features['context']= 0
    if type != 'op':
        if is_twocontext:
            features.update({'context_pro': 0, 'context_con': 0, 'question': 0})
        else:
            features.update({'context':0, 'question':0})
    for s in sets:
        for feature in features:
            if feature not in {'labels', 'ids', 'issues'}:
                if sentence_tok:
                    token = np.load(CL_PATH / f'tok_{s}_{feature}.npy')
                    ids = np.array([[[stoi[t] for t in s] for s in d] for d in token])
                else:
                    token = np.load(CL_PATH / f'tok_{s}_{feature}.npy')
                    ids = np.array([[stoi[t] for t in r] for r in token])

                assert len(ids) == len(token)
                np.save(CL_PATH / f'ids_{s}_{feature}.npy', ids)


def load_npys(path, pre_name, type=None):
    d = dict()
    if type == 'op':

        d[pre_name + "_label"] = np.load(path / (pre_name + '_labels.npy')).tolist()
        d[pre_name + "_opinion"] = np.load(path / (pre_name + '_opinion.npy')).tolist()

    else:
        d[pre_name + "_question"] = np.load(path / (pre_name + '_question.npy')).tolist()
        d[pre_name + "_opinion"] = np.load(path / (pre_name + '_opinion.npy')).tolist()
        d[pre_name + "_label"] = np.load(path / (pre_name + '_labels.npy')).tolist()
        d[pre_name + "_id"] = np.load(path / (pre_name + '_ids.npy')).tolist()
        #d[pre_name + "_int_id"] = np.load(path / (pre_name + '_int_ids.npy')).tolist()
        if type == "twocontext":
            d[pre_name + "_context_pro"] = np.load(path / (pre_name + '_context_pro.npy')).tolist()
            d[pre_name + "_context_con"] = np.load(path / (pre_name + '_context_con.npy')).tolist()

        elif type == "sent":
            d[pre_name + "_context"] = np.load(path / (pre_name + '_context.npy')).tolist()
            d[pre_name + "_bool_sent"] = np.load(path / (pre_name + '_bool_sent.npy')).tolist()
        elif type == "vader":
            d[pre_name + "_context"] = np.load(path / (pre_name + '_context.npy')).tolist()
            d[pre_name + "_vader_sent"] = np.load(path / (pre_name + '_vader_sent.npy')).tolist()
        else :
            d[pre_name + "_context"] = np.load(path / (pre_name + '_context.npy')).tolist()



    return d


def load_concat_npys(path, pre_name, type=None, is_twocontex=False, flat_sntc=False):
    if type == 'op':
        o = np.load(path / (pre_name + '_opinion.npy'))
        return o

    if is_twocontex:
        c_pro = np.load(path / (pre_name + '_context_pro.npy'))
        c_con = np.load(path / (pre_name + '_context_con.npy'))
        q = np.load(path / (pre_name + '_question.npy'))
        o = np.load(path / (pre_name + '_opinion.npy'))
        return np.concatenate((c_pro, c_con, q, o))

    elif flat_sntc:
        c = np.load(path / (pre_name + '_context.npy'))
        q = np.load(path / (pre_name + '_question.npy')).reshape(1,-1)[0]
        o = np.load(path / (pre_name + '_opinion.npy'))

        c = [np.concatenate(c[i]).tolist() for i in range(c.size)]
        o = [np.concatenate(o[i]).tolist() for i in range(o.size)]
        #o = [np.concatenate(o[i]).reshape(-1, 1) for i in range(o.size)]
        return np.concatenate((c, q, o))

    else:
        c = np.load(path / (pre_name + '_context.npy'))
        q = np.load(path / (pre_name + '_question.npy'))
        o = np.load(path / (pre_name + '_opinion.npy'))
        return np.concatenate((c, q, o))


def get_embed_matrix(embed_file, embed_sz, stoi):

    word_to_embed = get_word_to_embed(embed_file)
    embed_matrix = np.random.random((len(stoi), embed_sz))
    for word, i in stoi.items():
        if word in word_to_embed:
            embed_matrix[i] = word_to_embed[word]
    return embed_matrix


def get_word_to_embed(embedding_file):
    embeddings_index = {}
    f = open(embedding_file)
    for line in f:
        values = line.split()
        word, coef = values[0],  np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coef
    f.close()
    return embeddings_index


def softmax(vector):
    upper = [math.exp(v) for v in vector]
    sum_upper = sum(upper)
    return [u/sum_upper for u in upper]


def euclidean_distance(x1, x2, dim=1):
    r"""Returns Euclidean distance between x1 and x2, computed along dim.
        Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
    """
    return (torch.sqrt(torch.sum((x1-x2) ** 2, dim))).squeeze()


def mean_of_l1(x1, x2, dim=1):
    return torch.mean(torch.abs(x1 - x2), dim).squeeze()


def sigmoid_kernel(x1, x2, dim=1, gamma=None, c=1): # (batch,dim) , (batch, dim)
    dot = torch.bmm(x1.unsqueeze(1), x2.unsqueeze(2)).squeeze(-1)
    if gamma is None:
        gamma = 1.0/x1.size()[dim]
    output = torch.tanh(gamma * dot + c)
    return output


def chi_squared(x1, x2, dim=1, gamma=1, eps=1e-8):
    return torch.exp(- gamma * torch.sum(((x1 - x2) ** 2)/(x1 + x2).clamp(min=eps), dim))


def rbf_kernel(x1, x2, dim=1, gamma=1):
    output = torch.sum((x1 - x2) ** 2, dim)
    return torch.exp(- gamma * output)


def sim_func(l_hidden, r_hidden, dim=1):
    dot = torch.bmm(l_hidden.unsqueeze(1), r_hidden.unsqueeze(2)).squeeze(-1)
    cos = F.cosine_similarity(l_hidden, r_hidden, dim=dim).unsqueeze(1)
    euc = euclidean_distance(l_hidden, r_hidden, dim=dim).unsqueeze(1)
    mean_l1 = mean_of_l1(l_hidden, r_hidden, dim=dim).unsqueeze(1)
    sig = sigmoid_kernel(l_hidden, r_hidden, dim=dim)
    #chi = chi_squared(l_hidden, r_hidden, dim=dim).unsqueeze(1).clamp(-1.0, 1.0)
    rbf = rbf_kernel(l_hidden, r_hidden, dim=dim).unsqueeze(1)
    v = torch.cat([dot, cos, euc, mean_l1, sig,  rbf], dim=1)
    return v


def merge_arguments_quotes(root):
    with open(root / "arg_quot_train.csv", 'w') as fw:
        with open(root / "arguments_train.csv", "r") as fr:
            for line in fr:
                fw.write(line)
        with open(root / "quotes_train.csv", "r") as fr:
            for line in fr:
                fw.write(line)

    with open(root / "arg_quot_val.csv", 'w') as fw:
        with open(root / "arguments_val.csv", "r") as fr:
            for line in fr:
                fw.write(line)
        with open(root / "quotes_val.csv", "r") as fr:
            for line in fr:
                fw.write(line)

    with open(root / "arg_quot_tst.csv", 'w') as fw:
        with open(root / "arguments_tst.csv", "r") as fr:
            for line in fr:
                fw.write(line)
        with open(root / "quotes_tst.csv", "r") as fr:
            for line in fr:
                fw.write(line)


def save_checkpoint(model, is_best, model_name):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(model.state_dict(),  model_name)


def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def max_freq(idx_pool):
    maxes = []
    for l in idx_pool:
        l = torch.Tensor.numpy(l.cpu().squeeze()).tolist()
        freq = FreqDist(l)
        maxes.append(freq.most_common(1)[0][1])
    return max(maxes)


def plot_all(itos, idx_pool, seq, file_path='txt.jpeg', mx=10, fig_size=(10,20), pro_con_indices=None):
    idx_pool = torch.Tensor.numpy(idx_pool.cpu().squeeze()).tolist()
    freq = FreqDist(idx_pool)
    for i in range(len(idx_pool)):
        idx = idx_pool[i]
        f = freq[idx]
        if len(seq) < idx:
            print(len(seq))
        if not isinstance(seq[idx], tuple):
            seq[idx] = (seq[idx], f)
    for j in range(len(seq)):
        e = seq[j]
        if not isinstance(e, tuple):
            seq[j] = (seq[j], 0)

    lbls = [itos[e[0]] for e in seq]
    vals = [e[1] for e in seq]
    weights = None

    if pro_con_indices is not None:
        pro_inx, con_idx = pro_con_indices
        intro_weight = np.mean(vals[:pro_inx])
        pro_weight = np.mean(vals[pro_inx:con_idx])
        con_weight = np.mean(vals[con_idx:])
        weights = (intro_weight, pro_weight, con_weight)

    top_vals = list(sorted(vals, reverse=True))[:5]
    top_lbls = [lbls[i] for i in range(len(vals)) if vals[i] in top_vals]



    if lbls[0] == " ":
        lbls = lbls[1:]
        vals =vals[1:]

    n = int(len(lbls) / 20)
    if len(lbls) % 20 != 0:
        n = n + 1
    fig, axes = plt.subplots(nrows=n, figsize=fig_size)
    for i in range(0, n):
        m = min((i + 1) * 20, len(lbls))
        val_plot = vals[i * 20: m]
        lbl_plot = lbls[i * 20: m]
        if len(val_plot) < 20:
            k = 20 - len(val_plot)
            val_plot = val_plot + [0] * k
            lbl_plot = lbl_plot +[" "] * k
        # plotting (works fine)
        d = pd.DataFrame(np.array(val_plot).reshape(1, -1))

        ax = sns.heatmap(d, cmap="Reds", square=True, annot=True, xticklabels=lbl_plot, yticklabels=False,
                         ax=axes[i], vmax=mx, cbar=False)
        ax.set_xticklabels(rotation=40, labels=lbl_plot)

    fig.savefig(str(file_path))
    return weights, top_lbls


def to_glove_word_index(sent_tok_file, out_path):
    sentiments = np.load(sent_tok_file).tolist()
    vocabs = set([w for line in sentiments for w in line])
    word_index , index_word = dict(), dict()
    for  i, w in enumerate(vocabs):
        word_index[w] = i
        index_word [i] = w

    word_index['_default_'] = len(word_index)
    index_word[len(word_index)-1] = '_default_'
    assert len(word_index) == len(index_word)
    save_obj(word_index, out_path /'word_index_glove_sent.pkl' )
    save_obj(index_word, out_path / 'index_word_glove_sent.pkl')




def to_ids_glove_sent(path):
    train = np.load(path /'tok_train_opinion.npy')
    tst = np.load(path / 'tok_tst_opinion.npy')
    val = np.load(path / 'tok_val_opinion.npy')

    wtoi= load_obj(path /'word_index_glove_sent.pkl')

    train = [[wtoi[w] for w in line if w in wtoi] for line in train]
    train = np.array([t  if len(t)>0 else [wtoi['_default_']] for t in train])

    val = [[wtoi[w] for w in line if w in wtoi] for line in val]
    val = np.array([t if len(t) > 0 else [wtoi['_default_']] for t in val])

    tst = [[wtoi[w] for w in line if w in wtoi] for line in tst]
    tst = np.array([t if len(t) > 0 else [wtoi['_default_']] for t in tst])

    print(tst)

    np.save(path /'ids_train_glove_sent.npy', train)
    np.save(path / 'ids_val_glove_sent.npy', val)
    np.save(path / 'ids_tst_glove_sent.npy', tst)

def to_ids_bool_sent(path):
    train = np.load(path /'tok_train_opinion.npy')
    tst = np.load(path / 'tok_tst_opinion.npy')
    val = np.load(path / 'tok_val_opinion.npy')

    wtoi= load_obj(path /'word_index_glove_sent.pkl')

    train = [[1.0 if w in line else 0.0 for w in wtoi] for line in train]
    train = np.array([t  if len(t)>0 else [wtoi['_default_']] for t in train])

    val = [[1.0 if w in line else 0.0 for w in wtoi] for line in val]
    val = np.array([t if len(t) > 0 else [wtoi['_default_']] for t in val])

    tst = [[1.0 if w in line else 0.0 for w in wtoi] for line in tst]
    tst = np.array([t if len(t) > 0 else [wtoi['_default_']] for t in tst])

    for  e in val:
        print(np.sum(e))

    np.save(path /'ids_train_bool_sent.npy', train)
    np.save(path / 'ids_val_bool_sent.npy', val)
    np.save(path / 'ids_tst_bool_sent.npy', tst)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return d


def get_claims(claims, context, topic_id):
    l = []
    for c in claims:
        label = "1" if c["stance"] == "PRO" else "0"
        d = (str(c["claimId"]) + f"_{topic_id}", str(label), context, context, c["claimCorrectedText"])
        l.append(d)
    return l


def ibm_to_procon(ibm_path, procon_path):
    ibm = load_json(ibm_path)
    d = defaultdict(list)
    for i in range(len(ibm)):
        claims = get_claims(ibm[i]["claims"], ibm[i]["topicText"], ibm[i]["topicId"])
        s = "train_val" if ibm[i]["split"] == "train" else "tst"
        d[s].extend(claims)

    for s in d:
        df = pd.DataFrame(d[s], columns=["id", "label", "context", "question", "idea"])
        df.to_csv(procon_path / f'ibm_{s}.csv', index=False, header=False)
        print(s, len(d[s]))


def create_train_val_ibm(path):

    datasets = [pd.read_csv(path / 'ibm_train_val.csv', header=None)]
    train, val = [], []
    issues = defaultdict(dict)
    for ds in datasets:
        for i in range(1, len(ds[0])):
            iss = ds.iloc[i][0].split('_')[1]
            lbl = str(ds.iloc[i][1])
            if lbl in issues[iss].keys():
                issues[iss][lbl].append(ds.iloc[i])
            else:
                issues[iss][lbl] = [ds.iloc[i]]

    print(f'# of issues: {len(issues.keys())}, size of dataset :{len(ds[0])}')
    p, n = 0, 0
    for iss in issues.keys():
        v = issues[iss]
        print(f'{iss} +ve : {len(v["1"])}, avg -ve : {len(v["0"])}')
        tr_p, tr_n = int(.8 * len(v["0"])), int(.8 * len(v["1"]))
        train += v["0"][:tr_p] + v["1"][:tr_n]
        val += v["0"][tr_p:] + v["1"][tr_n:]

        p += len(v['1'])
        n += len(v['0'])
    print(f'+ve {p}, -ve :{n}, total :{p+n}, avg +ve : {p/len(issues)}, avg -ve: {n/len(issues)}')

    train = pd.concat([pd.DataFrame(t for t in train)])
    val = pd.concat([pd.DataFrame(t for t in val)])

    print(f' train: {len(train)}, val: {len(val)}, total = {len(train)+ len(val)}')

    print('train pos, neg : ', get_pos_neg(train))
    print('val pos, neg : ', get_pos_neg(val))

    val.to_csv(path / 'ibm_val.csv', header=False, index=False)
    train.to_csv(path / 'ibm_train.csv', header=False, index=False)


def get_pos_neg(df):
    p,n =[], []
    for i in range(len(df[0])):
        if df.iloc[i][1] == 0:
            n.append(i)
        elif df.iloc[i][1] == 1:
            p.append(i)
        else:
            print("Error in label!")
    return len(p),len(n)

def printPredsToFileByID(tar, pred, tarfile, predfile, dim=3):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """

    pred = np.argmax(pred, 1)
    assert len(tar) == len(pred)
    i=1000
    with open(tarfile,'w') as fw:
        fw.write("ID	Target	Tweet	Stance\n")
        for t in tar:
            t= "AGAINST" if t==0 else "FAVOR"
            fw.write(f"{i}\tblah\tblah\t{t}\n")
            i+=1

    i=1000
    with open(predfile,'w') as fw:
        fw.write("ID	Target	Tweet	Stance\n")
        for p in pred:
            p ="AGAINST" if p==0 else "FAVOR"
            fw.write(f"{i}\tblah\tblah\t{p}\n")
            i+=1

def eval(file_gold, file_pred, evalscript="eval.pl"):
    """
    Evaluate using the original script, needs to be in same format as train/dev data
    :param file_gold: testing file with gold standard data
    :param file_pred: file containing predictions
    :param evalscript: file location for official eval script
    """
    pipe = subprocess.Popen(["perl", evalscript, file_gold, file_pred], stdout=sys.stdout)
    pipe.communicate()

if __name__=="__main__":
    root = Path('/marjan/me/me/project/procon_ai/data/twitter')
    PATH_CLS = root / 'twitter_clas'
    root.mkdir(exist_ok=True)

    # IBM
    '''ibm_to_procon(root / 'claim_stance_dataset_v1.json', root)
    create_train_val_ibm(root)
    d = "ibm"
    save_tokens(root, d, type=None, features=None, ibm=True)
    save_ids(root, d, type=None, features=None)'''


    ds = ['twitter']


    for d in ds:
        #features = {'sentiment': 'sent_KL_top20.pkl'}
        save_tokens(root, d, type='twitter', features=None, sentence_tok=False, sets=['train', 'val', 'tst'])
        save_ids(root, d, type=None, features=None, sets=['train', 'val', 'tst'])

        #to_glove_word_index(PATH_CLS / 'tok_train_sentiment.npy', PATH_CLS)
        #to_ids_bool_sent(PATH_CLS)


        #save_tokens_twocontext(root, d, type=None, features=None)
        #save_ids(root, d, type=None, features=None, is_twocontext=True)

        #save_tokens(root, d, type=None, features=None, sentence_tok=True)
        #save_ids(root, d, type=None, features=None, sentence_tok=True)


