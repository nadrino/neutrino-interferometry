from nu_waves.models.mass import Mass
import numpy as np

print("Normal ordering (Δm31² > 0)")
spec = Mass(n=3)
spec.set_dm2({(2,1):7.4e-5, (3,1):2.5e-3})
spec.summary()

print("Inverted ordering (Δm32² < 0)")
spec = Mass(n=3)
spec.set_dm2({(2,1):7.4e-5, (3,2):2.43e-3})
spec.summary()

# print("Inverted ordering (Δm31² < 0)")
# spec = Spectrum(dm2={(2,1):7.4e-5, (3,1):-2.5e-3}, m_lightest=0.01)
# spec.summary()
#
# print("Normal ordering (Δm32² > 0)")
# spec = Spectrum(dm2={(2,1):7.4e-5, (3,2):2.43e-3}, m_lightest=0.01)
# spec.summary()


