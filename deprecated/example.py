import MMI_SParams as mmi
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


###HELPER FUNCTIONS####

def print_mode(mode_vector):
    for i, mode in enumerate(mode_vector):
        amplitude = np.abs(mode[0]) 
        phase = np.degrees(np.angle(mode[0]))
        print(f"Mode {i+1}: Amplitude = {amplitude:.4f}, Phase = {phase:.4f} degrees")


###EXAMPLE USAGE####

    #gets the sparams for a 2x2 MMI coupler at a wavelength of 1.55um
spararms = mmi.MMI_SMatrix(wavelength_um=1.55, example=True) #THIS IS HOW YOU GET THE SPARAMATERS

input_modes = np.array([[1], [0]])  # Example input mode (single mode excitation)
output_modes = (spararms @ input_modes)  # Apply the S-parameters to the input mode

print("Input Modes:")
print_mode(input_modes)

print("Output Modes:")
print_mode(output_modes)


####Graphic Picture of waht is going on####

    #example using a sinwave and the hilbert transform to get the output modes from the S-parameters
input_modes = np.array([(np.sin(np.linspace(0, 8*np.pi, 100))), np.zeros(100)])  # Example input mode (sine wave)
output_modes = (spararms @ sp.hilbert(input_modes))  # Apply the S-parameters to the input mode
plt.plot(np.real(input_modes.T), label='Input Mode')
plt.plot(np.real(output_modes.T), label='Output Modes')
plt.xlabel('Mode Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
