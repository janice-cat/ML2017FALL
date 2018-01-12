# -*- coding: utf-8 -*-
import jieba
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import gensim
import numpy as np
from sklearn.manifold import TSNE
from adjustText import adjust_text
import sys



def main():
	sentences = MySentences()
	#model = gensim.models.Word2Vec(sentences, size=80, iter=10)
	model = gensim.models.Word2Vec(sentences, size=80, iter=10, hs=1)
	#model.save('model/w2v_3') ###
	#model = gensim.models.Word2Vec.load('model/w2v_gensim')
	N = 6500
	Xs, Ys, Texts = get_coord(N, model)
	plot(Xs, Ys, Texts)



def get_coord(N, model):
	embed = []
	Texts = []
	for word, vocab_obj in model.wv.vocab.items():
		if (vocab_obj.count >= N):
			embed.append(model[word]) ## (100,)
			Texts.append(word)
	print(len(embed))
	embed = np.array(embed, dtype=np.float64)
	#perform t-SNE embedding
	vis_data = TSNE(n_components=2).fit_transform(embed)
	Xs = vis_data[:,0]
	Ys = vis_data[:,1]
	return(Xs, Ys, Texts)


def plot(Xs, Ys, Texts):
	myfont = FontProperties(fname = '/Library/Fonts/Arial Unicode.ttf')
	rcParams['axes.unicode_minus'] = False
	plt.plot(Xs, Ys, 'o')
	texts = [plt.text(X, Y, Text,fontproperties=myfont,fontsize=12) for X, Y, Text in zip(Xs, Ys, Texts)]
	plt.title(str(adjust_text(texts, Xs, Ys, arrowprops=dict(arrowstyle='->', color='red'))))
	plt.show()


class MySentences(object):
	jieba.set_dictionary('dict.txt.big')
	def __iter__(self):
		for line in open(sys.argv[1]):
			yield list(jieba.cut(line, cut_all=False))


if __name__ == '__main__':
	main()