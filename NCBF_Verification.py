import torch

from Verifier import Verifier
from NCBF_Synth import *
from Cases.CExample import CExample
CExample = CExample()
newCBF = NCBF_Synth([32, 32], [True, True], CExample, verbose=True)
newCBF.model.load_state_dict(torch.load('/Users/ericzhang/Documents/GitHub/FTNCBF/CE_1.pt'),strict=True)
veri_result, num = newCBF.veri.proceed_verification()
