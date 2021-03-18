import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from eps_eff import calculate_eps_eff

data_folder = "results\\mpb_test\\"

bandstructure = np.loadtxt(data_folder + "bandstructure.txt")

print(np.shape(bandstructure))

#Load all fields to numpy arrays

file_ref = h5py.File(data_folder + 'fielddata_band1_kpoint0.h5', 'r')
res = file_ref['efield_x_real'].shape[0]
print(res)
Efield_array = np.zeros((res,res,3,np.shape(bandstructure)[0],np.shape(bandstructure)[1]),dtype=complex)
Dfield_array = np.zeros((res,res,3,np.shape(bandstructure)[0],np.shape(bandstructure)[1]),dtype=complex)
Bfield_array = np.zeros((res,res,3,np.shape(bandstructure)[0],np.shape(bandstructure)[1]),dtype=complex)
Hfield_array = np.zeros((res,res,3,np.shape(bandstructure)[0],np.shape(bandstructure)[1]),dtype=complex)

for k_index in range(0,np.shape(bandstructure)[0]-1):
       for band_index in range(0,np.shape(bandstructure)[1]):
              file_ref = h5py.File(data_folder + 'fielddata_band' + str(band_index+1) + '_kpoint' + str(k_index) + '.h5')
              Efield_array[:, :, 0, k_index, band_index] = file_ref['efield_x_real'][0:res, 0:res]+1j*file_ref['efield_x_imag'][0:res, 0:res]
              Efield_array[:, :, 1, k_index, band_index] = file_ref['efield_y_real'][0:res, 0:res]+1j*file_ref['efield_y_imag'][0:res, 0:res]
              Efield_array[:, :, 2, k_index, band_index] = file_ref['efield_z_real'][0:res, 0:res]+1j*file_ref['efield_z_imag'][0:res, 0:res]

              Dfield_array[:, :, 0, k_index, band_index] = file_ref['dfield_x_real'][0:res, 0:res]+1j*file_ref['dfield_x_imag'][0:res, 0:res]
              Dfield_array[:, :, 1, k_index, band_index] = file_ref['dfield_y_real'][0:res, 0:res]+1j*file_ref['dfield_y_imag'][0:res, 0:res]
              Dfield_array[:, :, 2, k_index, band_index] = file_ref['dfield_z_real'][0:res, 0:res]+1j*file_ref['dfield_z_imag'][0:res, 0:res]

              Bfield_array[:, :, 0, k_index, band_index] = file_ref['bfield_x_real'][0:res, 0:res]+1j*file_ref['bfield_x_imag'][0:res, 0:res]
              Bfield_array[:, :, 1, k_index, band_index] = file_ref['bfield_y_real'][0:res, 0:res]+1j*file_ref['bfield_y_imag'][0:res, 0:res]
              Bfield_array[:, :, 2, k_index, band_index] = file_ref['bfield_z_real'][0:res, 0:res]+1j*file_ref['bfield_z_imag'][0:res, 0:res]

              Hfield_array[:, :, 0, k_index, band_index] = file_ref['hfield_x_real'][0:res, 0:res]+1j*file_ref['hfield_x_imag'][0:res, 0:res]
              Hfield_array[:, :, 1, k_index, band_index] = file_ref['hfield_y_real'][0:res, 0:res]+1j*file_ref['hfield_y_imag'][0:res, 0:res]
              Hfield_array[:, :, 2, k_index, band_index] = file_ref['hfield_z_real'][0:res, 0:res]+1j*file_ref['hfield_z_imag'][0:res, 0:res]

omegas = np.linspace(0.3, 0.5, 30)

epsilons = calculate_eps_eff(omegas,bandstructure[17:30,:],Efield_array[:,:,:,17:30,:],Bfield_array[:,:,:,17:30,:],Hfield_array[:,:,:,17:30,:],Dfield_array[:,:,:,17:30,:])

fig, ax = plt.subplots()
ax.plot(omegas,np.real(np.squeeze(epsilons[2,2,:])), 'o')

ax.set(xlabel='Energy \u03A9 a// 2\u03C0', ylabel=' Permittivity \u03B5'+ 'zz',
       title='Effective Permittivity vs. Energy')
ax.grid()
plt.show()
