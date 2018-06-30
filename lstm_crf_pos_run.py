import torch
import torch.autograd as autograd
import torch.nn as nn
import pickle
from nltk import word_tokenize
from nltk import pos_tag
import sys

print(sys.argv[1])

torch.manual_seed(1)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = map(lambda x: to_ix[x.lower()] if x in to_ix else to_ix['<unk>'], seq)
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, tag_count, tag_embedding_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeds = nn.Embedding(tag_count, tag_embedding_size)
        self.lstm = nn.LSTM(embedding_dim + tag_embedding_size, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.transitions.data[1][2] = -100000
        self.transitions.data[2][0] = -100000
        self.transitions.data[1][1] = -100000
        self.transitions.data[2][tag_to_ix[START_TAG]] = -100000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, sentence_pos):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        pos_embeds_out = self.pos_embeds(sentence_pos).view(len(sentence), 1, -1)
        tot_embedding = torch.cat((embeds, pos_embeds_out), dim=2)
        lstm_out, self.hidden = self.lstm(tot_embedding, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, sentence_pos, tags):
        feats = self._get_lstm_features(sentence, sentence_pos)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, sentence_pos):
        lstm_feats = self._get_lstm_features(sentence, sentence_pos)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

# load word2id
with open('word2id.pickle', 'rb') as f:
    word2id = pickle.load(f)

# load pos2id
with open('pos2id.pickle', 'rb') as f:
    pos2id = pickle.load(f)

# load the trained model
model = torch.load("best_model_lstm_crf_pos.pt")

sentence = sys.argv[1]

# tokenize the sentence
sentence = word_tokenize(sentence)

# keep original sentence tokens to get the output
original_sentence = sentence

# obtain the pos tags
sentence_pos = map(lambda tag: pos2id[tag] if tag in pos2id else pos2id['<unk>'], [x[1] for x in pos_tag(sentence)])
sentence_pos = torch.tensor(sentence_pos, dtype=torch.long)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

with torch.no_grad():
    sentence = prepare_sequence(sentence, word2id)
    _, tags = model(sentence, sentence_pos)

# print the aspect terms to the stdout
aspects = []

i = 0
while(i < len(sentence)):
    if tags[i] == 1:
        aspect = original_sentence[i]
        i += 1
        while(i < len(sentence) and tags[i]==2):
            aspect += " " + original_sentence[i]
            i += 1
        aspects.append(aspect)
    else:
        i += 1

print(aspects)