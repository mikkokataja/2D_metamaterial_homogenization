import numpy as np

def calculate_eps_eff(omegas,bandstructure,Efields,Bfields,Hfields,Dfields):
    epsilonsmatrix = np.zeros((6, 6, len(omegas)), dtype=complex)
    omega_i = 0
    for omega in omegas:
        Average_fields = list()
        for band_index in range(1, np.shape(bandstructure)[1]):
            for k_index in range(1, np.shape(bandstructure)[0] - 1):
                if bandstructure[k_index,band_index]-omega == 0:
                    Efield = np.squeeze(Efields[:, :, :, k_index, band_index])
                    Bfield = np.squeeze(Bfields[:, :, :, k_index, band_index])
                    Hfield = np.squeeze(Hfields[:, :, :, k_index, band_index])
                    Dfield = np.squeeze(Dfields[:, :, :, k_index, band_index])
                    Average_fields.append(wnbk_interpolant(Efield, Bfield, Hfield, Dfield))
                elif (bandstructure[k_index,band_index]-omega)*(bandstructure[k_index-1,band_index]-omega) < 0:
                    weightfactor = (omega-bandstructure[k_index-1,band_index])/(bandstructure[k_index,band_index]-bandstructure[k_index-1,band_index])
                    Efield = (1-weightfactor)*np.squeeze(Efields[:,:,:,k_index-1,band_index])+weightfactor*np.squeeze(Efields[:,:,:,k_index,band_index])
                    Bfield = (1-weightfactor)*np.squeeze(Bfields[:,:,:,k_index-1,band_index])+weightfactor*np.squeeze(Bfields[:,:,:,k_index,band_index])
                    Hfield = (1-weightfactor)*np.squeeze(Hfields[:,:,:,k_index-1,band_index])+weightfactor*np.squeeze(Hfields[:,:,:,k_index,band_index])
                    Dfield = (1-weightfactor)*np.squeeze(Dfields[:,:,:,k_index-1,band_index])+weightfactor*np.squeeze(Dfields[:,:,:,k_index,band_index])
                    Average_fields.append(wnbk_interpolant(Efield, Bfield, Hfield, Dfield))
        epsilonsmatrix[:, :, omega_i] = calculate_epseff(Average_fields)
        omega_i += 1

    return epsilonsmatrix

def wnbk_interpolant(Efield, Bfield, Hfield, Dfield):
    resolution = np.shape(Efield)[0]

    Eaverage = np.zeros_like(Efield)
    Baverage = np.zeros_like(Bfield)
    Haverage = np.zeros_like(Hfield)
    Daverage = np.zeros_like(Dfield)

    # Curl conforming fields

    w1 = 2 * np.matmul(np.ones((resolution, 1)), np.reshape(np.linspace(1, 0, resolution), (1, resolution)))
    w2 = 2 * np.matmul(np.ones((resolution, 1)), np.reshape(np.linspace(0, 1, resolution), (1, resolution)))
    w3 = 2 * np.matmul(np.reshape(np.linspace(1, 0, resolution), (resolution, 1)), np.ones((1, resolution)))
    w4 = 2 * np.matmul(np.reshape(np.linspace(0, 1, resolution), (resolution, 1)), np.ones((1, resolution)))

    w9 = np.matmul(np.reshape(np.linspace(1, 0, resolution), (resolution, 1)),
                   np.reshape(np.linspace(1, 0, resolution), (1, resolution)))
    w10 = np.matmul(np.reshape(np.linspace(0, 1, resolution), (resolution, 1)),
                    np.reshape(np.linspace(1, 0, resolution), (1, resolution)))
    w11 = np.matmul(np.reshape(np.linspace(1, 0, resolution), (resolution, 1)),
                    np.reshape(np.linspace(0, 1, resolution), (1, resolution)))
    w12 = np.matmul(np.reshape(np.linspace(0, 1, resolution), (resolution, 1)),
                    np.reshape(np.linspace(0, 1, resolution), (1, resolution)))

    Eaverage[:, :, 0] = w1 * np.mean(Efield[:, 1, 0]) + w2 * np.mean(Efield[:, -1, 0])
    Eaverage[:, :, 1] = w3 * np.mean(Efield[1, :, 1]) + w4 * np.mean(Efield[-1, :, 1])
    Eaverage[:, :, 2] = w9 * Efield[1, 1, 2] + w10 * Efield[-1, 1, 2] + w11 * Efield[1, -1, 2] + w12 * Efield[-1, -1, 2]
    
    Haverage[:, :, 0] = w1 * np.mean(Hfield[:, 1, 0]) + w2 * np.mean(Hfield[:, -1, 0])
    Haverage[:, :, 1] = w3 * np.mean(Hfield[1, :, 1]) + w4 * np.mean(Hfield[-1, :, 1])
    Haverage[:, :, 2] = w9 * Hfield[1, 1, 2] + w10 * Hfield[-1, 1, 2] + w11 * Hfield[1, -1, 2] + w12 * Hfield[-1, -1, 2]

    # Div conforming fields

    v1 = np.matmul(np.reshape(np.linspace(1, 0, resolution), (resolution, 1)), np.ones((1, resolution)))
    v2 = np.matmul(np.reshape(np.linspace(0, 1, resolution), (resolution, 1)), np.ones((1, resolution)))
    v3 = np.matmul(np.ones((resolution, 1)), np.reshape(np.linspace(1, 0, resolution), (1, resolution)))
    v4 = np.matmul(np.ones((resolution, 1)), np.reshape(np.linspace(0, 1, resolution), (1, resolution)))

    Daverage[:, :, 0] = v1 * np.mean(Dfield[1, :, 0]) + v2 * np.mean(Dfield[-1, :, 0])
    Daverage[:, :, 1] = v3 * np.mean(Dfield[:, 1, 1]) + v4 * np.mean(Dfield[:, -1, 1])
    Daverage[:, :, 2] = np.mean(Dfield[:, :, 2])

    Baverage[:,:, 0] = v1 * np.mean(Bfield[1, :, 0]) + v2 * np.mean(Bfield[-1, :, 0])
    Baverage[:,:, 1] = v3 * np.mean(Bfield[:, 1, 1]) + v4 * np.mean(Bfield[:, -1, 1])
    Baverage[:,:, 2] = np.mean(Bfield[:, :, 2])

    average_fields = [Eaverage, Baverage, Haverage, Daverage]
    return average_fields

def calculate_epseff(average_fields):
    if not average_fields:
        epsmatrix = np.zeros((6, 6), dtype=complex)
    else:
        resolution = np.shape(average_fields[0][0])[0]
        num_bands = len(average_fields)
        epsmatrix_aux = np.zeros((resolution, resolution, 6, 6), dtype=complex)
        for index_x in range(0, resolution):
            for index_y in range(0, resolution):
                E_vector = np.zeros((6, num_bands), dtype=complex)
                D_vector = np.zeros((6, num_bands), dtype=complex)
                for band_index in range(0, num_bands):
                    E_vector[:, band_index] = np.concatenate([average_fields[band_index][0][index_x, index_y, :], average_fields[band_index][2][index_x, index_y, :]])
                    D_vector[:, band_index] = np.concatenate([average_fields[band_index][3][index_x, index_y, :], average_fields[band_index][1][index_x, index_y, :]])
                epsmatrix_aux[index_x, index_y, :, :] = np.matmul(D_vector, np.linalg.pinv(E_vector))
        epsmatrix = np.mean(epsmatrix_aux, axis = (0, 1))
    return epsmatrix



