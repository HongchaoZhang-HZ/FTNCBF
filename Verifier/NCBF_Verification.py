from Verifier import *

newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
newCBF.model.load_state_dict(torch.load(sys.path[-1]+'/'+'NCBF.pt'))
# Define Case
x0, x1 = sp.symbols('x0, x1')

# hx = (x[0] + x[1] ** 2)
hx = (x0 + x1**2)
# x0dot = x1 + 2*x0*x1
# x1dot = -x0 + 2*x0**2 - x1**2
fx = [x1 + 2*x0*x1,-x0 + 2*x0**2 - x1**2]
gx = [0, 0]
Darboux = case(fx, gx, hx, newCBF.DOMAIN, [])

veri = Verifier(NCBF=newCBF, case=Darboux, grid_shape=[100, 100], verbose=True)
verify_res = veri.proceed_verification()