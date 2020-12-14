from spelling_model import *
from s2s_transformer_model import *
from encoder_decoder_model import EDSpellModel
from encoder_decoder_attention_model import EDASpellModel
from simple_ed_model import *
from s2s_transformer_model import SpellingTransformer

#mclass = EDSpellModel #SpellingAttention #SpellingTransformer
#mclass = SEDASpellingModel
#mclass = EDSpellModel
#mclass = SpellingModel
#mclass = SEDASpellingModel
#mclass = SpellingTransformer
mclass = ECCNNSpellingModel

max_len = 20
#fname = 'data/sentences.txt'
fname = 'bare_kps.txt'
all_phrases = load_preprocessed(fname, max_len)
all_phrases = all_phrases[:20_000]
model = mclass(max_len, 128)
model.BATCH_SIZE = 500
model.init_from_texts(all_phrases)

model.train(all_phrases, 200, val_size=2_000)

model.model.summary()

model.report(all_phrases[:200])

phrases = all_phrases[:100]

miss_phrases = [add_noise_to_string(p, .05) for p in phrases]
