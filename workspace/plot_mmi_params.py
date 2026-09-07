import numpy as np
import matplotlib.pyplot as plt
data = np.load("14-15um.npz")
data2 = np.load("15-16um.npz")
#join the structured arrays row-wise
# data = np.load("2um.npz")
smatricies = np.concatenate((data['smatricies'], data2['smatricies']))
wavelengths = np.concatenate((data['wavelengths'], data2['wavelengths']))
print(wavelengths.shape)
print(np.diff(wavelengths))


S31 = smatricies[:, 0, 0]
S41 = smatricies[:, 1, 0]

s31Mag = np.abs(S31)
s31Phase = np.angle(S31, deg=True)

s41Mag = np.abs(S41)
s41Phase = np.angle(S41, deg=True)



def IdealGroupDelay(length, wavelengths, lambda0, phi0):
    f = 1/wavelengths
    group_delay = ((length * f + phi0) % 360) - 360    
    return f, group_delay

f = 1/wavelengths
plt.scatter(f, s31Phase, label='∠S31', color='blue')
f0, group_delay = IdealGroupDelay(283341, wavelengths, 1.55, -182955)
plt.scatter(f, group_delay, label='Ideal Group Delay', color='green')
plt.show()


plt.plot(wavelengths, np.unwrap(s31Phase - s41Phase, discont=180), label='∠S31-S41', color='blue')
plt.title('Phase Difference between S31 and S41')
plt.xlabel('Frequency (THz)')
plt.ylabel('Phase Difference (degrees)')
plt.legend()
plt.show()

plt.scatter(wavelengths, 20*np.log10(s31Mag) - 20*np.log10(s41Mag), label='Imbalance', color='blue')
plt.legend()
plt.show()




plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.scatter(wavelengths, 20*np.log10(s31Mag), label='|S31|', color='blue')
plt.scatter(wavelengths, 20*np.log10(s41Mag), label='|S41|', color='orange')
plt.title('Magnitude of S31')
plt.xlabel('Frequency (THz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(wavelengths, s31Phase, label='∠S31', color='blue')
plt.scatter(wavelengths, s41Phase, label='∠S41', color='orange')
# plt.scatter(wavelengths, np.unwrap(s31Phase-s41Phase, discont=180), label='∠S31-S41', color='blue')
plt.title('Phase of S31')
plt.xlabel('Frequency (THz)')
plt.ylabel('Phase (degrees)')
plt.legend()

plt.tight_layout()
plt.show()