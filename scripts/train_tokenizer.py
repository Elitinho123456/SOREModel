import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

caminho_dados_corpus = ['data/corpus_completo.json']

tokenizer_bpe = Tokenizer(BPE())
tokenizer_bpe.pre_tokenizer = Whitespace()

bpeTrainer = BpeTrainer(vocab_size=20000, show_progress=True)

tokenizer_bpe.train(caminho_dados_corpus, trainer=bpeTrainer)
print("Tokenizador treinado com sucesso!")

core_dir = 'src/core/tokenizer'

save_path = os.path.join(core_dir, "sore_bpe_tokenizer.json")
tokenizer_bpe.save(save_path)

print(f"Tokenizer BPE salvo em: {save_path}")