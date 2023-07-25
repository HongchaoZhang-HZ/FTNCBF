from Verifier import *
from NCBF_Synth import *
from Cases import ObsAvoid
ObsAvoid = ObsAvoid()
newCBF = NCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
veri_result, num = newCBF.veri.proceed_verification()