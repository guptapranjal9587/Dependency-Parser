from gensim.scripts.glove2word2vec import glove2word2vec
import os


def main() :
	if not os.path.isfile("gensim_glove_vectors_50d.txt") :
		print("Converting Glove input file to gensim Word2Vec file format...")
		glove2word2vec(glove_input_file = "glove.6B.50d.txt", word2vec_output_file = "gensim_glove_vectors_50d.txt")

	#glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors_100d.txt", binary = False)
	#modelVocab = list(glove_model.wv.vocab)
	#print("Glove Word2Vec model loaded...")

if __name__ == "__main__" :
	main()
