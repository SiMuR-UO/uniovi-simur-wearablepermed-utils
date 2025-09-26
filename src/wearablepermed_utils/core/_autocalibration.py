import numpy as np
import scipy as sp
import statsmodels.api as sm

from . import _preprocessing as prep
from . import _feature_extraction as feat

def count_stuck_vals(xArray, yArray, zArray, stuckVal = 1.5, clip_value = 8):
    """
    Counts the number of stuck values in the given x, y, and z arrays.

    A value is considered "stuck" if the standard deviation of the array is zero 
    and the mean of the array is outside the range [-1.5, 1.5]. Additionally, if 
    the maximum absolute value in any of the arrays is 8, it is considered stuck.

    Parameters:
    xArray (numpy.ndarray): Array of x values.
    yArray (numpy.ndarray): Array of y values.
    zArray (numpy.ndarray): Array of z values.

    Returns:
    int: The number of stuck values detected.
    """
    #get necessary background stats...
    xMean = np.mean(xArray)
    yMean = np.mean(yArray)
    zMean = np.mean(zArray)
    
    xStd = np.std(xArray)
    yStd = np.std(yArray)
    zStd = np.std(zArray)
    
    #see if values are likely to have been abnormally stuck during this epoch
    numStuckValues = 0
    
    if ((xStd == 0) and ((xMean < -stuckVal) or (xMean > stuckVal))) or \
       ((yStd == 0) and ((yMean < -stuckVal) or (yMean > stuckVal))) or \
       ((zStd == 0) and ((zMean < -stuckVal) or (zMean > stuckVal))):
        numStuckValues = len(zArray) #it could be also len(yArray) or len(xArray), they should be of the same length
        
    if np.max(np.abs(xArray))>=clip_value or np.max(np.abs(yArray))>=clip_value or np.max(np.abs(zArray))>=clip_value:
        numStuckValues = 1
        
    return numStuckValues

def get_calibration_coefs(caldata, MAXITER = 1000, IMPROV_TOL = 0.0001, ERR_TOL = 0.015, CALIB_CUBE = 0.3, CALIB_MIN_SAMPLES = 50): 
    """
    Calculate calibration coefficients for a given set of calibration data.
    Parameters:
    caldata (dict): Dictionary containing calibration data with key 'xyzMean' which is a numpy array of shape (3, N).
    MAXITER (int, optional): Maximum number of iterations for the calibration process. Default is 1000.
    IMPROV_TOL (float, optional): Improvement tolerance for stopping criteria. Default is 0.0001.
    ERR_TOL (float, optional): Error tolerance for determining good calibration. Default is 0.015.
    CALIB_CUBE (float, optional): Threshold for checking uniformly distributed points. Default is 0.3.
    CALIB_MIN_SAMPLES (int, optional): Minimum number of samples required for calibration. Default is 50.
    Returns:
    tuple: A tuple containing:
        - bestIntercept (numpy array): Intercept coefficients for the calibration.
        - bestSlope (numpy array): Slope coefficients for the calibration.
        - bestSlopeT (numpy array): SlopeT coefficients for the calibration.
        - bestErr (float): The best error achieved during the calibration process.
    """
    xyz = caldata['xyzMean']
    T = np.zeros(xyz.shape[1])

    # Remove any zero vectors as they cause nan issues
    nonzero = sp.linalg.norm(xyz, axis=0) > 1e-8
    xyz = xyz[:,nonzero]
    T = T[nonzero]

    intercept = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    slopeT = np.array([0.0, 0.0, 0.0])
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestSlopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=0)
    errors = np.linalg.norm(curr - target, axis=0)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    initErr = err
    bestErr = 1e16
    
    # Check that we have enough uniformly distributed points:
    # need at least one point outside each face of the cube
    if xyz.shape[1] < CALIB_MIN_SAMPLES:
        goodCalibration = 0
    elif (np.max(xyz, axis=1) < CALIB_CUBE).any():
        goodCalibration = 0
    elif (np.min(xyz, axis=1) > -CALIB_CUBE).any():
        goodCalibration = 0
    else:  # we do have enough uniformly distributed points
        for it in range(MAXITER):
            # Weighting. Outliers are zeroed out
            # This is different from the paper
            maxerr = np.quantile(errors, .995)
            weights = np.maximum(1 - errors / maxerr, 0)
            # Optimize params for each axis
            for k in range(3):
                inp = curr[k, :]                 # Cambiado de curr[:, k] a curr[k, :], para obtener la fila con todos los puntos
                out = target[k, :]               # Cambiar también target[:, k] a target[k, :]
                inp = np.column_stack((inp, T))  # Ahora inp tiene la misma longitud que T
                inp = sm.add_constant(inp, prepend=True, has_constant='add')
                params = sm.WLS(out, inp, weights=weights).fit().params
                intercept[k] = params[0] + (intercept[k] * params[1])
                slope[k] = params[1] * slope[k]
                slopeT[k] = params[2] + (slopeT[k] * params[1])
            # Update current solution and target
            curr = intercept[:, np.newaxis] + (xyz * slope[:, np.newaxis]) # Es importante utilizar la opción np.newaxis
            curr = curr.T + (slopeT.T * T[:, None])
            target = curr / np.linalg.norm(curr, axis=1, keepdims=True)
            # Update errors
            errors = np.linalg.norm(curr - target, axis=1)
            err = np.mean(errors)
            errImprov = (bestErr - err) / bestErr
            if err < bestErr:
                bestIntercept = np.copy(intercept)
                bestSlope = np.copy(slope)
                bestSlopeT = np.copy(slopeT)
                bestErr = err
            if errImprov < IMPROV_TOL:
                break
            curr=curr.T        # Matriz traspuesta de curr, IMPORTANTE!
            target=target.T    # Matriz traspuesta de target, IMPORTANTE!
        goodCalibration = int(not ((bestErr > ERR_TOL) or (it + 1 == MAXITER)))
    if goodCalibration == 0:  # restore calibr params
        bestIntercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
        bestSlope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
        bestSlopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
        bestErr = initErr

    return bestIntercept, bestSlope, bestSlopeT, bestErr

def auto_calibrate(acc_data, window_size = 10, fm = 25, clipping_value = 8):
    """
    Perform automatic calibration of acceleration data.
    Parameters:
    acc_data (numpy.ndarray): The raw acceleration data to be calibrated. 
                                Expected to be a 2D array where rows represent 
                                samples and columns represent [timestamp, x, y, z].
    window_size (int, optional): The size of the window for epoch segmentation in seconds. 
                                    Default is 10 seconds.
    fm (int, optional): The sampling frequency in Hz. Default is 25 Hz.
    Returns:
    numpy.ndarray: The calibrated acceleration data.
    """

    # Clipping de los datos de aceleración
    prep.clip_data(acc_data, [1,2,3], clipping_value)
    
    Tm=1/fm*1000   # Período de muestreo, expresado en [ms]
    
    # Interpolación de los datos
    interpolados = prep.time_interp(acc_data, Tm, 0)
        
    tArray=interpolados[:,0]                          # Obtenemos el vector de tiempos (timestamps) interpolados
    duracion=tArray[-1]-tArray[0]                     # Duración del vector temporal
    duracion_de_epocas= window_size*1000                        # Duración de cada época, expresada en [ms]
    muestras_en_epoca= window_size*fm                           # Número de muestras por época (10 [s] de la ventana temporal con una frecuencia de muestreo de 25 [Hz])
    epocas=int(np.floor(duracion/duracion_de_epocas)) # 10 segundos por defecto
    calData={}                                        # Diccionario de datos calibrados
    calData['xyzMean']=np.zeros((3,epocas))           # Valores medios de los datos de aceleración, en sus 3 componentes cartesianas
    calData['stuckValues']=np.zeros(epocas)           # En esta entrada del diccionario, se guardan valores asociados a fallos en el registro del IMU.
    for k in range(0, epocas):
        #Tomamos los datos de cada epoca
        epochtime=tArray[k*muestras_en_epoca:muestras_en_epoca*(k+1)+1]
        epochdata=interpolados[k*muestras_en_epoca:muestras_en_epoca*(k+1)+1, 1:4]
        xArray=epochdata[:,0]
        yArray=epochdata[:,1]
        zArray=epochdata[:,2]
    
        # print(str(epochtime[0])+'-'+str(epochtime[-1]))
        basic_statistics, _, _ = feat.get_basic_stats(epochdata[:, 1:])
        #print(basic_statistics)
    
        errCounter = count_stuck_vals(xArray, yArray, zArray)
    
        #means = basic_statistics[2]
        calData['xyzMean'][0,k]=basic_statistics[2]
        calData['xyzMean'][1,k]=basic_statistics[3]
        calData['xyzMean'][2,k]=basic_statistics[4]
        calData['stuckValues'][k]=errCounter
    
    # Obtención coeficientes de la calibración: offset, slope y error.
    offset, slope, slopeT, err= get_calibration_coefs(calData) 
    print(f'offset:{offset}, slope:{slope}, err:{err}')
    
    datos_acc_calibrados = acc_data[:,1:4] * slope + offset
    
    return datos_acc_calibrados, slope, offset