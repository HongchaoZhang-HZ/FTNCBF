from NNCBF_Synth_V1.Verifier.NCBF_Verification import *

def main():
    newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
    newCBF.train(4)
    visualization(newCBF)

if __name__ == '__main__':
    newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
    newCBF.train(1)
    poly_name, poly_coeff = newCBF.topolyCBF()
