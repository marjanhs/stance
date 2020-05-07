US = "\x1f"  # unit separator => sentence separator
soh = "\x02"
import sys
import subprocess

sys.path.insert(0, "/fastai")
from fastai.text import *
from procon_ai_utils import *
from procon_ai_mydataloader import MyDataLoader
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', type=int, default=10, help='batch size')
    parser.add_argument('-hd_sz', type=int, default=1150, help='hidden size')
    parser.add_argument('-embd_sz', type=int, default=400, help='embedding size')
    parser.add_argument('-dir', type=int, default=1, help='lstm direction')
    parser.add_argument('-n_layers', type=int, default=3, help='# of layer')
    parser.add_argument('-iscuda', type=bool, default=True, help='gpu')
    parser.add_argument('-lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('-embed_file', type=str, default=None, help='glove wordtovec')
    # parser.add_argument('-embed_file', type=str, default='../glove.6B.100d.txt', help='glove wordtovec')

    return parser.parse_args()


class MyRNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, qrnn=False):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.

            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs, self.qrnn, self.emb_sz = 1, qrnn, emb_sz
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                1, bidirectional=bidir) for l in range(n_layers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz,self.n_hid,self.n_layers,self.dropoute = emb_sz,n_hid,n_layers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(n_layers)])
        #self.word_context_vector = self.init_word_contx_vector()
        #self.lin_attention = nn.Linear(self.emb_sz, self.emb_sz) #  rnn[-1] : (n_hid, emb_sz)


    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        #print(input.size())
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
            #self.word_context_vector = self.init_word_contx_vector()
        with set_grad_enabled(self.training):
            emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
            emb = self.dropouti(emb)
            raw_output = emb
            new_hidden,raw_outputs,outputs = [],[],[]
            for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1: raw_output = drop(raw_output)
                #else: raw_output = self.get_word_attention(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def init_word_contx_vector(self):
        return nn.Parameter(torch.Tensor(1, self.bs, self.emb_sz).uniform_(-1.0, 1.0)).cuda() #changed

    def get_word_attention(self, word_encoded):
        # word_encoded :(sl, bs, hs), word_context_vector:(1, bs,  hs)
        a = self.lin_attention(word_encoded)
        u = torch.tanh(a)  # (sl, bs, hs)
        mul = (u * self.word_context_vector).sum(-1)  # (sl, bs)
        alpha = F.softmax(mul, dim=0).unsqueeze(2)  # (sl,bs) -> (sl,bs, 1), for each word we have one aplha
        # we want to have one vector for one sentence, so we sum up all weighted word-vectors, first dim is words
        #s = (word_encoded * alpha).sum(0).squeeze(0)  # (1,bs,hs) -> (bs,hs)
        s = (word_encoded * alpha).sum(0).unsqueeze(0) # (1,bs,hs)
        return s

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        if IS_TORCH_04: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        if self.qrnn: [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.n_layers)]


class MyMultiBatchRNN(MyRNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        raw_outputs, outputs = [],[]

        #print(f'sl: {sl}, max_seq:{self.max_seq}')
        sl = min(self.max_seq, sl)
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

        '''for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)'''


class ParallelULM(nn.Module):
    def __init__(self, args):
        super(ParallelULM, self).__init__()
        self.opinion_model = MyMultiBatchRNN(args.bptt, args.max_seq, args.n_tok, args.embd_sz, args.hd_sz, args.n_layers,
                                           pad_token=args.pad_token, bidir=args.bidir,
                                           dropouth=args.dps[0], dropouti=args.dps[1], dropoute=args.dps[2],
                                           wdrop=args.dps[3], qrnn=False)

        self.context_model = MyMultiBatchRNN(args.bptt, args.max_seq, args.n_tok, args.embd_sz, args.hd_sz, args.n_layers,
                                           pad_token=args.pad_token, bidir=args.bidir,
                                           dropouth=args.dps[0], dropouti=args.dps[1], dropoute=args.dps[2],
                                           wdrop=args.dps[3], qrnn=False)

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

    def forward(self, *inputdata):
        inputdata = inputdata[0]
        context, opinion = inputdata[0], inputdata[1]
        raw_context, out_context = self.context_model(context.cuda())
        raw_opinion, out_opinion = self.opinion_model(opinion.cuda())
        #out_merged = [torch.cat((out_context[-1], out_opinion[-1]), dim=0)]  # (sl,bs,emb_sz)
        #raw_merged = [torch.cat((raw_context[-1], raw_opinion[-1]), dim=0)]  # (sl,bs,emb_sz)

        out_merged = [out_context[-1], out_opinion[-1]]  # (sl,bs,emb_sz)
        raw_merged = [raw_context[-1], raw_opinion[-1]]  # (sl,bs,emb_sz)
        return raw_merged, out_merged


class MyTextDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        item = self.x[idx]
        context = np.array(item[0])
        opinion = np.array(item[1])
        label = item[-1]
        return context, opinion, label

    def __len__(self): return len(list(self.x))


class MySampler(Sampler):  # returns the indices of the dataset with respect to context's length descending
    def __init__(self, dataset, key):
        self.dataset, self.key = dataset, key

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(sorted(range(len(self.dataset)), key=self.key, reverse=True))

class MyLinearBlock(nn.Module):
    def __init__(self, nil, nf, drop, bias=False):
        super().__init__()
        self.lin = nn.Linear(nil, nf, bias)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(nil)
        self.bias = torch.nn.Parameter(torch.Tensor(1))
        self.reset_bias()

    def reset_bias(self):
        if self.bias is not None:
            self.bias.data.uniform_(-1, 1)

    def forward(self, x):

        return self.lin(self.drop(self.bn(x)))


class MyPoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(0, len(layers) - 1)])


    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        r = f(x.permute(1,2,0), (1,)).view(bs,-1)
        return r

    def forward(self, input):
        raw_output_context, raw_output_opin = input[0]
        output_context, output_opin = input[1]

        raw_outputs = torch.cat([raw_output_context, raw_output_opin], 0) #???
        outputs = torch.cat([output_context, output_opin], 0) # ?????

        sl,bs,_ = output_context.size()
        avgpool_c = self.pool(output_context, bs, False)
        mxpool_c = self.pool(output_context, bs, True)
        x_context = torch.cat([output_context[-1], mxpool_c, avgpool_c], 1)



        sl, bs, _ = output_opin.size()
        avgpool_o = self.pool(output_opin, bs, False)
        mxpool_o = self.pool(output_opin, bs, True)
        x_opin = torch.cat([output_opin[-1], mxpool_o, avgpool_o], 1)


        x = torch.cat([x_context, x_opin], 1)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x, raw_outputs, outputs


def my_get_rnn_classifier(args, layers, drops):
    ulm_enc = ParallelULM(args).cuda()
    return MySequentialRNN(ulm_enc, MyPoolingLinearClassifier(layers, drops))


class MySequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

    def forward(self, *input):
        for module in self._modules.values():
            input = module(input)
        return input


class ACCScheduler(Callback):
    def __init__(self, learn):
        self.learn = learn

    def on_train_begin(self):
        self.first_epoch = True
        self.best_acc = 0

    def on_epoch_end(self, metrics):
        val_acc = metrics[0]
        if self.first_epoch:
            self.best_acc = val_acc
            self.first_epoch = False
        elif val_acc > self.best_acc and val_acc > 0.50:
            self.best_acc = val_acc
            self.learn.save('best_acc')
        else:
            return False


class SaveBestModel(LossRecorder):
    def __init__(self, model, lr, name='best_model'):
        super().__init__(model.get_layer_opt(lr, None))
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss, acc = metrics
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')


def my_main(PATH=None, ds=None, train_lm=False, train_model=False):
    print(f'Excecuting on  {ds}')
    LM_PATH = PATH / (ds + '_clas')
    LM_PATH.mkdir(exist_ok=True)

    OUT_PATH = PATH / (ds + '_clas') / 'out'
    OUT_PATH.mkdir(exist_ok=True)

    args = get_args()
    for arg in vars(args):
        print(f'{arg}={getattr(args, arg)}')

    # loading itos of  procon dataset

    args.bptt, em_sz, nh, nl = 70, 400, 1150, 3
    # 20 * 70
    #args.max_seq, args.pad_token = 20 * 70, 1  # rnns only returns the output of  the last max_seq of a seq
    args.max_seq, args.pad_token = 5 * 70, 1  # rnns only returns the output of  the last max_seq of a seq
    print(f'max seq len: {args.max_seq},  pre_pad= False')
    args.bidir = True if args.dir == 2 else False
    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    args.n_tok = len(itos)
    print(f'procon vocab size = {args.n_tok}')

    # loading wikitext LM
    PRE_PATH = PATH / 'models' / 'wt103'
    PRE_LM_PATH = PRE_PATH / 'fwd_wt103.h5'
    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)
    itos2 = pickle.load((PRE_PATH / 'itos_wt103.pkl').open('rb'))
    stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
    print(f'wikitext vocab size = {len(itos2)}')

    # creating new language model (lm1, lm1_enc) for procon dataset using wikitext pre-trained LM

    if train_lm:
        new_w = np.zeros((args.n_tok, args.embd_sz), dtype=np.float32)
        '''for i, w in enumerate(itos):
            r = stoi2[w]
            new_w[i] = enc_wgts[r] if r >= 0 else row_m'''

        wgts['0.encoder.weight'] = T(new_w)
        wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
        wgts['1.decoder.weight'] = T(np.copy(new_w))
        trn_lm = load_concat_npys(LM_PATH, 'ids_train')
        val_lm = load_concat_npys(LM_PATH, 'ids_val')
        wd = 1e-7
        bptt = 70
        bs = 52
        opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
        trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
        val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
        md = LanguageModelData(PATH, 1, args.n_tok, trn_dl, val_dl, bs=bs, bptt=bptt)
        drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.95
        learner = md.get_model(opt_fn, em_sz, nh, nl,
                               dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

        learner.metrics = [accuracy]
        learner.freeze_to(-1)
        learner.model.load_state_dict(wgts)
        lr = 1e-3
        lrs = lr
        learner.fit(lrs / 2, 1, wds=wd, use_clr=(32, 2), cycle_len=1)
        learner.unfreeze()
        learner.lr_find(start_lr=lrs / 10, end_lr=lrs * 10, linear=True)
        #learner.sched.plot()
        learner.fit(lrs, 1, wds=wd, use_clr=(20, 10), cycle_len=5)
        learner.save('lm')
        learner.save_encoder('lm_enc')
        #learner.sched.plot_loss()

    # loading train, val ,test ids and labels,  creating classifier

    trn_clas = load_npys(LM_PATH, 'ids_train')
    val_clas = load_npys(LM_PATH, 'ids_val')
    tst_clas = load_npys(LM_PATH, 'ids_tst')

    trn_input = list(zip(trn_clas["ids_train_context"], trn_clas["ids_train_opinion"],
                         trn_clas["ids_train_label"]))
    val_input = list(zip(val_clas["ids_val_context"], val_clas["ids_val_opinion"],
                         val_clas["ids_val_label"]))
    tst_input = list(zip(tst_clas["ids_tst_context"], tst_clas["ids_tst_opinion"],
                         tst_clas["ids_tst_label"]))
    ds_trn = MyTextDataset(trn_input)
    sa_trn = MySampler(ds_trn, key=lambda x: len(ds_trn[x][0]))
    dl_trn = MyDataLoader(ds_trn, args.bs, transpose=True, num_workers=1, pad_idx=1, sampler=sa_trn, pre_pad=False)

    ds_val = MyTextDataset(val_input)
    sa_val = MySampler(ds_val, key=lambda x: len(ds_val[x][0]))
    dl_val = MyDataLoader(ds_val, args.bs, transpose=True, num_workers=1, pad_idx=1, sampler=sa_val, pre_pad=False)

    ds_tst = MyTextDataset(tst_input)
    sa_tst = MySampler(ds_tst, key=lambda x: len(ds_tst[x][0]))
    dl_tst = MyDataLoader(ds_tst, args.bs, transpose=True, num_workers=1, pad_idx=1, sampler=sa_tst, pre_pad=False)

    args.ntoken = len(itos)
    if args.embed_file is not None:
        args.embd_matrix = get_embed_matrix(args.embed_file, args.embd_sz, stoi)
    else:
        args.embd_matrix = None

    c = 2
    rate = 1
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * rate
    args.dps = dps
    m = my_get_rnn_classifier(args=args, layers=[args.embd_sz * 6, 50, c], drops=[dps[4], 0.1])
    md = ModelData(PATH, dl_trn, dl_val, dl_tst)
    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    tm = MyTextModel(to_gpu(m))
    learn = MyRNN_Learner(md, tm, opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip = 25.
    learn.metrics = [accuracy]
    #acc_sched = ACCScheduler(learn)

    if train_model:
        lr = 3e-3
        lrm = 2.6
        #lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])
        #lrs = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])
        lrs = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-2]) #org
        #lrs = np.array([1e-2 for _ in range(9)])
        wd = 1e-7
        #wd = 0
        my_cb = SaveBestModel(learn, lrs, name='best_model')
        learn.load_encoder('lm_enc')
        learn.freeze_to(-1)
        learn.lr_find(lrs / 1000)
        #learn.sched.plot()
        learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))

        learn.freeze_to(-7)  # -1 +(-1)*3 *2  = -7
        learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))

        learn.unfreeze()
        learn.fit(lrs, 1, wds=wd, cycle_len=30, use_clr=(32, 10), callbacks=[my_cb])
        learn.sched.plot_loss()
        learn.save('last_model')
        #learn.sched.plot_loss()

    # best model
    model_name = 'best_model'
    learn.load('best_model')
    pred, tar = learn.predict_with_targs(is_test=True)
    acc = accuracy_np(pred, tar)
    print(f'model: {model_name}, Acc: {acc}')
    printPredsToFileByID(tar, pred, OUT_PATH / (model_name +'.tar'), OUT_PATH / (model_name +'.pred'))
    eval( OUT_PATH / (model_name +'.tar'),  OUT_PATH / (model_name +'.pred'))

    #last_model
    model_name = 'last_model'
    print(model_name)
    learn.load(model_name)
    pred, tar = learn.predict_with_targs(is_test=True)
    acc = accuracy_np(pred, tar)
    print(f'model: {model_name}, Acc: {acc}')
    printPredsToFileByID( tar, pred, OUT_PATH / (model_name +'.tar'), OUT_PATH / (model_name +'.pred'))
    eval(OUT_PATH / (model_name + '.tar'), OUT_PATH / (model_name + '.pred'))


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

def myload_model(m, p, emb_sz):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()):  # list "detatches" the iterator
        if n not in names and n + '_raw' in names:
            if n + '_raw' not in sd: sd[n + '_raw'] = sd[n]
            del sd[n]

        elif n not in names:
            sd['opinion_model.' + n] = sd[n]
            sd['context_model.' + n] = sd[n]
            del sd[n]
    m.load_state_dict(sd)


class MyRNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy

    def fit(self, *args, **kwargs): return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))

    def load_encoder(self, name): myload_model(self.model[0], self.get_model_path(name), self.models.model[0].opinion_model.emb_sz)


class MyTextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.opinion_model.encoder, m.opinion_model.dropouti),
                (m.context_model.encoder, m.context_model.dropouti),
                *zip(m.opinion_model.rnns, m.opinion_model.dropouths),
                *zip(m.context_model.rnns, m.context_model.dropouths),
                (self.model[1])]


if __name__ == "__main__":

    np.random.seed(1337)
    torch.cuda.manual_seed_all(1377)

	PATH = Path('data')
	print(str(PATH))
	results = []
	# fine-tune the language model:
	my_main(PATH, 'arg_quot', train_lm=True, train_model=False)
	
	#train the whole model	
	my_main(PATH, 'arg_quot', train_lm=False, train_model=True)





