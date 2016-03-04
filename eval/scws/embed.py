import argparse

import scikits.bootstrap as bootstrap

# from scipy.stats import entropy

from eval.scws.scws import Dataset
from eval.scws.correlation import spearman
from eval.scws.corrstats import independent_corr
from bimu.utils.generic_utils import *
from bimu.utils.logger import log
from bimu.utils.test_utils import cosine, avg_sim, softmax


def loc_predict(inst, embs, c_embs, bias, w_index, win):
    cs_1, cs_2 = inst.contexts(w_index, win=win)  # list of context word ids
    if not cs_1 or not cs_2:
        return None, None
    cmean_1 = np.mean(c_embs[np.array(cs_1)], axis=0)
    cmean_2 = np.mean(c_embs[np.array(cs_2)], axis=0)

    if bias is not None:
        act1 = np.dot(embs[inst.w1_idx], cmean_1) + bias[inst.w1_idx]
        act2 = np.dot(embs[inst.w2_idx], cmean_2) + bias[inst.w2_idx]
    else:
        act1 = np.dot(embs[inst.w1_idx], cmean_1)
        act2 = np.dot(embs[inst.w2_idx], cmean_2)

    return act1, act2  # n_senses


def local_emb(inst, embs, c_embs, bias, w_index, win):
    act1, act2 = loc_predict(inst, embs, c_embs, bias, w_index, win)
    s1 = np.argmax(np.exp(act1))
    s2 = np.argmax(np.exp(act2))

    if s1 is None:
        return None, None

    return embs[inst.w1_idx, s1], embs[inst.w2_idx, s2]


def avg_exp(inst, embs, c_embs, bias, w_index, win):
    act1, act2 = loc_predict(inst, embs, c_embs, bias, w_index, win)
    #s1 = softmax(act1)
    #s2 = softmax(act2)

    #print("{}".format(entropy(s1)))
    #print("{}".format(entropy(s2)))

    if act1 is not None:
        avg_emb1 = np.average(embs[inst.w1_idx], weights=softmax(act1), axis=0)
    else:
        avg_emb1 = np.mean(embs[inst.w1_idx], axis=0)

    if act2 is not None:
        avg_emb2 = np.average(embs[inst.w2_idx], weights=softmax(act2), axis=0)
    else:
        avg_emb2 = np.mean(embs[inst.w2_idx], axis=0)

    return avg_emb1, avg_emb2


def avg_emb(inst, embs):
    return np.mean(embs[inst.w1_idx], axis=0), np.mean(embs[inst.w2_idx], axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ci", help="Whether to measure confidence intervals.", default=False)
    parser.add_argument("-input_dir", help="Directory containing model and vocabulary files.", required=True)
    parser.add_argument("-input_dir2", help="Directory to compare with.")
    parser.add_argument("-f", default="W_w", help="Model file (optional), meant for models per epoch.")
    parser.add_argument("-data_path", help="Filepath containing the SCWS dataset.", default="data/SCWS/ratings.txt")
    parser.add_argument("-downcase", help="Use when the model was trained on downcased corpus.", default=False)
    parser.add_argument("-embs_format", default="bimu", choices=["bimu", "w2v"],
                        help="Format of embedding file and vocabulary.")
    parser.add_argument("-weight_type", default="pivot", choices=["pivot", "context", "shared"],
                        help="Whether to use pivot/context/shared embeddings.")
    parser.add_argument("-win_size", default=3, type=int,
                        help="Context window size (n words to the left and n to the right).")
    parser.add_argument("-model", choices=["sg", "senses"], default="sg", help="Model type being loaded.")
    parser.add_argument("-model2", choices=["sg", "senses"], default="sg", help="Model type being loaded.")
    parser.add_argument("-n_most_freq", type=int, help="Only consider n most freq. words from vocabulary.")
    parser.add_argument("-sim", choices=["local", "avg", "avg_emb", "avg_exp"], default="avg")
    args = parser.parse_args()
    #np.random.seed(1234)

    if args.embs_format == "w2v":
        w_index, embs = load_w2v("{}/vec.txt".format(args.input_dir))
        print("Loaded model.")
    #elif args.embs_format == "mssg":  # Neelakantan et al.'s multisense embs
    #    w_index, embs = load_mssg("{}/vectors-MSSGKMeans.gz".format(args.input_dir))
    #    if args.n_most_freq:
    #        w_index = {w: i for w, i in w_index.items() if i < args.n_most_freq + 1}
    #        print(len(w_index))
    #    print("Loaded model.")
    elif args.embs_format == "bimu":
        try:
            w_index_path = "{}/w_index.json".format(args.input_dir)
            w_index = load_json(w_index_path)
        # model_path = "{}/sg.pickle".format(args.input_dir)
        except FileNotFoundError:
            w_index_path = "{}/W_v.txt".format(args.input_dir)
            w_index = {l.strip(): c for c, l in enumerate(line_reader(w_index_path))}
        log.info("Loading model.")

        if args.n_most_freq:
            w_index = {w: i for w, i in w_index.items() if i < args.n_most_freq + 1}
            print(len(w_index))
        if args.model == "senses":
            embs = load_npy("{}/{}.npy".format(args.input_dir, args.f))
            if args.sim == "local" or args.sim == "avg_exp":
                if args.f == "W_w":
                    n = ""
                else:
                    n = eval(args.f[-1])
                    assert 0 <= n < 9
                c_embs = load_npy("{}/W_c{}.npy".format(args.input_dir, n))
            try:
                if args.f == "W_w":
                    n = ""
                else:
                    n = eval(args.f[-1])
                    assert 0 <= n < 9
                bias = load_npy("{}/Wb{}.npy".format(args.input_dir, n))
            except FileNotFoundError:
                bias = None
        else:
            if args.weight_type == "pivot":
                model_path = "{}/{}.npy".format(args.input_dir, args.f)
            elif args.weight_type == "context":
                model_path = "{}/W_c.npy".format(args.input_dir)
            elif args.weight_type == "shared":
                model_path = "{}/W.npy".format(args.input_dir)
            else:
                sys.exit("Unkown weight type.")
            embs = load_npy(model_path)
            print("Loaded model.")
        if args.input_dir2:
            try:
                w_index_path2 = "{}/w_index.json".format(args.input_dir2)
                w_index2 = load_json(w_index_path2)
            # model_path = "{}/sg.pickle".format(args.input_dir)
            except FileNotFoundError:
                w_index_path2 = "{}/W_v.txt".format(args.input_dir2)
                w_index2 = {l.strip(): c for c, l in enumerate(line_reader(w_index_path2))}
            log.info("Loading model2.")
            if args.n_most_freq:
                w_index2 = {w: i for w, i in w_index2.items() if i < args.n_most_freq + 1}
                print(len(w_index2))
            if args.model2 == "senses":
                embs2 = load_npy("{}/{}.npy".format(args.input_dir2, args.f))
                if args.sim == "local" or args.sim == "avg_exp":
                    if args.f == "W_w":
                        n = ""
                    else:
                        n = eval(args.f[-1])
                        assert 0 <= n < 9
                    c_embs2 = load_npy("{}/W_c{}.npy".format(args.input_dir2, n))
                try:
                    if args.f == "W_w":
                        n = ""
                    else:
                        n = eval(args.f[-1])
                        assert 0 <= n < 9
                    bias2 = load_npy("{}/Wb{}.npy".format(args.input_dir2, n))
                except FileNotFoundError:
                    bias2 = None
            else:
                if args.weight_type == "pivot":
                    model_path2 = "{}/{}.npy".format(args.input_dir2, args.f)
                elif args.weight_type == "context":
                    model_path2 = "{}/W_c.npy".format(args.input_dir2)
                elif args.weight_type == "shared":
                    model_path2 = "{}/W.npy".format(args.input_dir2)
                else:
                    sys.exit("Unkown weight type.")
                embs2 = load_npy(model_path2)
                print("Loaded model2.")
    else:
        sys.exit("Unrecognized embeddings format.")

    log.info("Loading dataset.")
    d = Dataset()
    d.create(args.data_path, w_index, downcase=args.downcase)

    rats = d.get_avg_rats()
    idxs1 = d.get_w1_idxs()
    idxs2 = d.get_w2_idxs()

    log.info("Obtaining embeddings.")
    rel_rats = []  # relevant average ratings

    if args.model == "senses" and args.sim == "avg":
        scores = []
        for inst in d:
            if inst.w1_idx is None:
                continue
            if inst.w2_idx is None:
                continue
            scores.append(avg_sim(embs[inst.w1_idx], embs[inst.w2_idx]))
            rel_rats.append(inst.avg_rat)
    else:
        rel_embs1 = []  # relevant embeddings of w1
        rel_embs2 = []  # relevant embeddings of w2
        for inst in d:
            if inst.w1_idx is None:
                continue
            if inst.w2_idx is None:
                continue
            if args.model == "senses":
                if args.sim == "local":
                    e1, e2 = local_emb(inst, embs, c_embs, bias, w_index, args.win_size)
                elif args.sim == "avg_emb":
                    e1, e2 = avg_emb(inst, embs)
                elif args.sim == "avg_exp":
                    e1, e2 = avg_exp(inst, embs, c_embs, bias, w_index, args.win_size)
            else:
                e1 = embs[inst.w1_idx]
                e2 = embs[inst.w2_idx]
            rel_embs1.append(e1)
            rel_embs2.append(e2)
            rel_rats.append(inst.avg_rat)

        log.info("Calculating distances and correlation.")
        assert len(rel_embs1) == len(rel_embs2) == len(rel_rats)
        scores = []
        for e1, e2 in zip(rel_embs1, rel_embs2):
            scores.append(cosine(e1, e2))

    assert len(scores) == len(rel_rats)

    if args.input_dir2:
        log.info("Loading dataset.")
        d2 = Dataset()
        d2.create(args.data_path, w_index2, downcase=args.downcase)

        rats2 = d2.get_avg_rats()
        idxs2_1 = d2.get_w1_idxs()
        idxs2_2 = d2.get_w2_idxs()

        log.info("Obtaining embeddings.")
        rel_rats2 = []  # relevant average ratings

        if args.model2 == "senses" and args.sim == "avg":
            scores2 = []
            for inst in d2:
                if inst.w1_idx is None:
                    continue
                if inst.w2_idx is None:
                    continue
                scores2.append(avg_sim(embs2[inst.w1_idx], embs2[inst.w2_idx]))
                rel_rats2.append(inst.avg_rat)
        else:
            rel_embs2_1 = []  # relevant embeddings of w1
            rel_embs2_2 = []  # relevant embeddings of w2
            for inst in d2:
                if inst.w1_idx is None:
                    continue
                if inst.w2_idx is None:
                    continue
                if args.model2 == "senses":
                    if args.sim == "local":
                        e1, e2 = local_emb(inst, embs2, c_embs2, bias2, w_index2, args.win_size)
                    elif args.sim == "avg_emb":
                        e1, e2 = avg_emb(inst, embs2)
                    elif args.sim == "avg_exp":
                        e1, e2 = avg_exp(inst, embs2, c_embs2, bias2, w_index2, args.win_size)
                else:
                    e1 = embs2[inst.w1_idx]
                    e2 = embs2[inst.w2_idx]
                rel_embs2_1.append(e1)
                rel_embs2_2.append(e2)
                rel_rats2.append(inst.avg_rat)

            log.info("Calculating distances and correlation.")
            assert len(rel_embs2_1) == len(rel_embs2_2) == len(rel_rats2)
            scores2 = []
            for e1, e2 in zip(rel_embs2_1, rel_embs2_2):
                scores2.append(cosine(e1, e2))

        assert len(scores2) == len(rel_rats2)

    corr = spearman(scores, rel_rats)
    log.debug("{} embedded words found out of {}.".format(len(scores), len(d)))
    log.info("Correlation: {0[0]}, p-value: {0[1]}.".format(corr))
    if args.ci:
        ci = bootstrap.ci((scores, rel_rats), statfunction=spearman, method="pi")
        log.info("CI: {0[0]} ({1}), {0[1]} (+{2}).".format(ci[:, 0], ci[:, 0][0] - corr[0], ci[:, 0][1] - corr[0]))
    if args.input_dir2:
        corr2 = spearman(scores2, rel_rats2)
        log.debug("Model2: {} embedded words found out of {}.".format(len(scores2), len(d)))
        log.info("Model2: Correlation: {0[0]}, p-value: {0[1]}.".format(corr2))
        if args.ci:
            ci2 = bootstrap.ci((scores2, rel_rats2), statfunction=spearman, method="pi")
            log.info("Model2: CI: {0[0]}, {0[1]}.".format(ci2[:, 0], ci2[:, 0][0] - corr2[0], ci2[:, 0][1] - corr2[0]))
        #corr_between = spearman(scores, scores2)
        #log.info("Between-models: Correlation: {0[0]}, p-value: {0[1]}.".format(corr_between))
        #sign = dependent_corr(corr[0], corr2[0], corr_between[0], n=len(rel_rats), twotailed=True, conf_level=0.95)
        #log.info("Significance: Test score: {0[0]}, p-value: {0[1]}.".format(sign))
        sign = independent_corr(corr[0], corr2[0], n=len(rel_rats), n2=len(rel_rats2), twotailed=False, conf_level=0.95)
        log.info("Significance: Test score: {0[0]}, p-value: {0[1]}.".format(sign))