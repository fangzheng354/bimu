import sys


class Conll07Reader:
    ### read Conll 2007 data: reusing https://github.com/bplank/myconllutils
    ### http://ilk.uvt.nl/conll/index.html#dataformat

    def __init__(self,filename):
        self.filename = filename
        self.startReading()

    def __iter__(self):
        i = self.getNext()
        while i:
            yield i
            i = self.getNext()

    def startReading(self):
        self.FILE = open(self.filename,"r")

    def getNext(self):
        # return next instance or None

        line = self.FILE.readline()

        line = line.strip()
        lineList = line.split("\t")

        ids = []
        form = []
        lemma = []
        cpos = []
        pos = []
        feats = []
        head = []
        deprel = []
        phead = []
        pdeprel = []

        if len(lineList) >= 12: #CONLL 2009 format
            while len(lineList) >= 12:
                ids.append(int(lineList[0]))
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[5])
                pos.append(lineList[4])
                feats.append(lineList[6])
                head.append(int(lineList[8]))
                deprel.append(lineList[10])
                phead.append(lineList[9])
                pdeprel.append(lineList[11])

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        elif len(lineList) == 10:
            # contains all cols, also phead/pdeprel
            while len(lineList) == 10:
                ids.append(int(lineList[0]))
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[3])
                pos.append(lineList[4])
                feats.append(lineList[5])
                head.append(int(lineList[6]))
                deprel.append(lineList[7])
                phead.append(lineList[8])
                pdeprel.append(lineList[9])

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        elif len(lineList) == 8:
            while len(lineList) == 8:
                ids.append(lineList[0])
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[3])
                pos.append(lineList[4])
                feats.append(lineList[5])
                head.append(int(lineList[6]))
                deprel.append(lineList[7])
                phead.append("_")
                pdeprel.append("_")

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        # conll2013 ner:
        elif len(lineList) == 9:
            while len(lineList) == 9:
                ids.append(lineList[2])
                form.append(lineList[5])
                lemma.append(lineList[5])
                cpos.append(lineList[0]) #ne
                pos.append(lineList[4])
                feats.append("_")
                head.append("_")
                deprel.append("_")
                phead.append("_")
                pdeprel.append("_")

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")

        # supersense tagging
        elif len(lineList) == 3:
            while len(lineList) == 3:
                ids.append("_")
                form.append(lineList[0])
                lemma.append(lineList[0])
                cpos.append(lineList[2]) # supersense
                pos.append(lineList[1]) # pos
                feats.append("_")
                head.append("_")
                deprel.append("_")
                phead.append("_")
                pdeprel.append("_")

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")

        elif len(lineList) > 1:
            raise Exception("not in right format!")

        if len(form) > 0:
            return DependencyInstance(ids,form,lemma,cpos,pos,feats,head,deprel,phead,pdeprel)
        else:
            return None


    def getInstances(self):
        instance = self.getNext()

        instances = []
        while instance:
            instances.append(instance)

            instance = self.getNext()
        return instances

    def getSentences(self):
        """ return sentences as list of lists """
        instances = self.getInstances()
        sents = []
        for i in instances:
            sents.append(i.form)
        return sents

    def getStrings(self, wordform="form"):
        """ sentence is one space-separated string in a list """
        if wordform=="lemma":
            return (" ".join(instance.lemma) for instance in self)
        else:
            return (" ".join(instance.form) for instance in self)

    def writeStrings(self, filepath, wordform="form"):
        """ write form to output. """
        with open(filepath, "w") as out:
            for i in self.getStrings(wordform=wordform):
                out.write("{}\n".format(i))

    def getVocabulary(self, n_sent=float("Inf"), add_root=True, lemmas=False):
        """
         vocabulary with frequencies
         :param n_sent: max number of sentences to consider
         :param add_root: add artificial symbol *root* to vocab
         :param lemmas: use lemma instead of form
        """
        from collections import defaultdict
        vocab = defaultdict(int)
        instance = self.getNext()
        c = 1
        if lemmas:
            while instance and (c<=n_sent):
                for w in instance.getSentenceLemmas():
                    vocab[w] += 1
                vocab["*root*"] += 1
                instance = self.getNext()
                c += 1
        else:
            while instance and (c<=n_sent):
                for w in instance.getSentence():
                    vocab[w] += 1
                vocab["*root*"] += 1
                instance = self.getNext()
                c += 1
        return vocab

    def getRelationVocabulary(self, n_sent=float("Inf")):
        """
         vocabulary of relation labels
         :param n_sent: max number of sentences to consider
         :param add_root: add artificial symbol *root* to vocab
        """
        vocab = set()
        instance = self.getNext()
        c = 1
        while instance and (c<=n_sent):
            vocab.update(instance.deprel)
            instance = self.getNext()
            c += 1
        return vocab

    def getCorpusTriples(self, wordform="form"):
        """ gets counts of head_w\tdep_w occurences """
        from collections import defaultdict
        counts = defaultdict(int)

        if wordform == "form":
            for instance in self:
                for i in instance.getBareFormTriples():
                    counts[i] += 1
        elif wordform == "lemma":
            for instance in self:
                for i in instance.getBareLemmaTriples():
                    counts[i] += 1
        return counts

    def getkCorpusTriples(self, k):
        """ gets counts of head_w\tdep_w occurences for k instances """
        from collections import defaultdict
        counts = defaultdict(int)

        for instance in self:
            if k > 0:
                for i in instance.getBareFormTriples():
                    counts[i] += 1
                k -= 1
            else:
                break
        return counts

def writeCorpusTriples(counts, filepath):
    with open(filepath, "w") as out:
        for k,v in counts.items():
            out.write("{0[0]}\t{0[1]}\t{1}\n".format(k.split("\t"),v))

class DependencyInstance:

    def __init__(self, ids, form, lemma, cpos, pos, feats, headid, deprel, phead, pdeprel):
        self.ids = ids
        self.form = form
        self.lemma = lemma
        self.cpos = cpos
        self.pos = pos
        self.feats = feats
        self.headid = headid
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n"
        sout = ""
        for i in range(len(self.form)):
            sout += s.format(self.ids[i],self.form[i],self.lemma[i],self.cpos[i],self.pos[i],self.feats[i],self.headid[i],self.deprel[i],self.phead[i],self.pdeprel[i])
        return sout

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield (self.ids[i],self.form[i],self.lemma[i],self.cpos[i],self.pos[i],self.feats[i],self.headid[i],self.deprel[i],self.phead[i],self.pdeprel[i])


    def writeout(self, filepath):
        if self is not None:
            with open(filepath, "a") as out:
                out.write(self.__str__())
                out.write("\n")

    def writeout_handle(self, fileh):
        if self is not None:
            fileh.write(self.__str__())
            fileh.write("\n")

    def equalForm(self,instance):
        for f1,f2 in zip(self.form,instance.form):
            if f1 != f2:
#            if f1 != "<num>" and f2 != "<num>" and f1 != f2:
                return False
        return True

    def equalHeads(self,instance):
        for f1,f2 in zip(self.headid,instance.headid):
            if f1 != f2:
                return False
        return True

    def containsRelation(self,label):
        return label in self.deprel

    def containsWord(self,word):
        return word in self.form

    def containsPostag(self,postag):
        return postag in self.pos

    #def positionPostag(self, postag):
       # return self.pos.index(self,postag)

    def equalLabels(self,instance):
        for f1,f2 in zip(self.deprel,instance.deprel):
            if f1 != f2:
                return False
        return True

    def equalHeadsAndLabels(self,instance):
        return self.equalHeads(instance) and self.equalLabels(instance)

    def getSentenceLength(self):
        return len(self.form)

    def getSentence(self):
        return self.form

    def getIds(self):
        return self.ids

    def getHeads(self):
        """
         head of id
        """
        return zip(self.ids, self.headid)

    def getSentenceLemmas(self):
        return self.lemma

    def getSentenceByPos(self):
        sentence = []
        for i in range(len(self.pos)):
            if self.pos[i] in ["NOM", "NAM", "ABR", "VER", "PRP", "NUM", "ADV", "ADJ", "VER:subp","VER:futu", "VER:simp", "VER:subi","VER:pres", "VER:cond","VER:ppre", "VER:impf"]:
                sentence.append(self.lemma[i])
        return sentence

    def getLemmaTriples(self):
        return self.getTriples(self.lemma)

    def getFormTriples(self):
        return self.getTriples(self.form)

    def getBareFormTriples(self):
        return self.getBareTriples(self.form)

    def getBareLemmaTriples(self):
        return self.getBareTriples(self.lemma)

    def getBareTriples(self,wordform):
        """ no counts. no rel. tab-sep. no root triple"""
        for i in range(len(wordform)):
            w_d = wordform[i]
            hid = self.headid[i]
            if hid != 0:
                w_h = wordform[hid-1]
                if type(w_h) != str:
                    print(wordform)
            else:
                #w_h = '<root-LEMMA>'
                continue
            yield "{}\t{}".format(w_h,w_d)

    def getTriples(self,wordform):
        triples = {}
        for i in range(len(wordform)):
            r = self.deprel[i]
            w_d = wordform[i].replace(" ","")
            hid = self.headid[i]
            if hid != 0:
                w_h = wordform[hid-1].replace(" ","")
            else:
                w_h = '<root-LEMMA>'
            triple = "{} {} {}".format(r,w_h,w_d)
            triples[triple] = triples.get(triple,0) + 1
        #triples = self._addExtendedTriples(triples,wordform)
        return triples

    def getAllLemmaTriples(self):
        return self.getAllTriples(self.lemma)

    def getAllFormTriples(self):
        return self.getAllTriples(self.form)

    def getAllTriples(self,wordform):
        """ also returns counts of parts of relation """
        triples = self.getTriples(wordform)
        actualtriples = triples.copy()
        for triple in actualtriples:
            try:
                r,w_h,w_d = triple.split(" ")

                #triple_r_w1 = "{} {} _".format(r,w_h)
                triple_r_w1 = "{} {} ".format(r,w_h)
                triples[triple_r_w1] = triples.get(triple_r_w1,0) + 1

                # Gertjan
                #triple_w2 = "_ _ {}".format(w_d)
                #triples[triple_w2] = triples.get(triple_w2,0) + 1

                #triple_w2x = "{} _ {}".format(r,w_d)
                triple_w2x = "{} {}".format(r,w_d)
                triples[triple_w2x] = triples.get(triple_w2x,0) + 1

                # Lin:
                #triple_r = "{} _ _".format(r)
                #triples[triple_r] = triples.get(triple_r,0) +1

            except ValueError:
                print("Error when splitting triples: {}".format(triple))
                sys.exit(-1)
        return triples

def filter_freq(s, f, vocab, lemma=False):
    if s is not None:
        for i in range(len(s)):
            if lemma:
                if vocab[s.lemma[i]] < f:
                    s.lemma[i] = "*unk*"
            else:
                if vocab[s.form[i]] < f:
                    s.form[i] = "*unk*"
        return s
    return

def filter_len(s, min_len, max_len):
    if s is not None:
        if min_len < len(s) < max_len:
            return s
    return
