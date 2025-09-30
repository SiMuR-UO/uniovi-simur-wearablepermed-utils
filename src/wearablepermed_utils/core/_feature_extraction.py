import numpy as np
import scipy as sp
from . import _preprocessing as prep
from scipy.signal import find_peaks
from numpy import arctan2, atan2, sqrt

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"
WINDOW_ALL_METADATA = "arr_2"


def get_basic_stats(epochdata, filter_b = [], filter_a = []):
    """
    Calculate basic statistics for the given epoch data.

    Parameters:
    epochdata (numpy.ndarray): A 2D array where each column represents a signal and each row represents a temporal value.
    filter_b (numpy.ndarray): The numerator coefficient vector of the filter.
    filter_a (numpy.ndarray): The denominator coefficient vector of the filter.

    Returns:
    tuple: A tuple containing the basic statistics, truncated ENMO, and filtered ENMO.
    """
    means = np.mean(epochdata, axis=0)
    ranges = np.ptp(epochdata, axis=0)
    cov_matrix = np.cov(epochdata, rowvar=False) #rowvar=False significa que cada columna de epochdata representa una variable diferente y cada fila es una observación.
    std_devs = np.sqrt(np.diag(cov_matrix))
    covariances = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]

    # Calculate ENMO
    enmo = prep.ENMO(epochdata)
    if len(filter_b) and len(filter_a):
        enmo_filtered = sp.signal.lfilter(filter_b, filter_a, enmo)
    else:
        enmo_filtered = enmo
        
    enmo_trunc = enmo_filtered * (enmo_filtered > 0)
    enmo_trunc_mean = np.mean(enmo_trunc)
    enmo_abs = np.abs(enmo_filtered)
    enmo_abs_mean = np.mean(enmo_abs)

    basic_statistics = [enmo_trunc_mean, enmo_abs_mean] + means.tolist() + ranges.tolist() + std_devs.tolist() + covariances.tolist()
    return basic_statistics, enmo_trunc, enmo_filtered

def get_FFT_power(FFT, normalize=True):
    """
    Calculate the power of the FFT (Fast Fourier Transform) coefficients.

    Parameters:
    FFT (array-like): The FFT coefficients.
    normalize (bool, optional): If True, normalize the power by the square of the length of the FFT. Default is True.

    Returns:
    array-like: The power of the FFT coefficients, optionally normalized.
    """
    n = len(FFT)
    FFTpow = FFT * np.conjugate(FFT)
    FFTpow = FFTpow.real
    if normalize:
        FFTpow = FFTpow / (n * n)
    return FFTpow

def obtener_caracteristicas_espectrales(v, fm):
    n = len(v)
    vMean = np.mean(v)
    vFFT = v - vMean
    vFFT = vFFT * np.hanning(n)
    
    # Realizar FFT
    vFFT = np.fft.rfft(vFFT)
    vFFTpow = get_FFT_power(vFFT)
    
    # Encontrar las frecuencias dominantes
    FFTinterval = fm / (1.0 * n)  # Resolución en Hz
    f1_idx = np.argmax(vFFTpow)   # Índice del máximo de potencia
    p1 = vFFTpow[f1_idx]          # Potencia máxima
    f1 = f1_idx * FFTinterval     # Frecuencia en Hz
    
    # Descartamos el primer pico para encontrar el siguiente
    vFFTpow[f1_idx] = 0  
    f2_idx = np.argmax(vFFTpow)  # Índice del segundo máximo de potencia
    p2 = vFFTpow[f2_idx]         # Potencia del segundo pico
    f2 = f2_idx * FFTinterval    # Frecuencia en Hz
    
    # Cálculo de la entropía espectral
    vFFTpowsum = np.sum(vFFTpow)                                # Suma total de las potencias FFT
    p = vFFTpow / (vFFTpowsum + 1e-8)                           # Probabilidades normalizadas
    spectralEntropy = np.sum(-p * np.log10(p + 1E-8))           # Entropía espectral
    spectralEntropy = spectralEntropy / np.log10(len(vFFTpow))  # Normalizamos la entropía
    
    return [f1, p1, f2, p2, spectralEntropy], vFFTpow

def get_FFT_magnitude(FFT, normalize=True):
    """
    Calculate the magnitudes from FFT (Fast Fourier Transform) coefficients.

    Parameters:
    FFT (array-like): The FFT coefficients.
    normalize (bool, optional): If True, normalize the magnitudes by the length of the FFT. Default is True.

    Returns:
    array-like: The magnitudes of the FFT coefficients.
    """
    # Use numpy's absolute function to get magnitudes from FFT coefficients
    FFTmag = np.abs(FFT)
    if normalize:
        FFTmag /= len(FFT)
    return FFTmag

def extract_features(data):
    # ***************
    # 1.- Cuantiles *
    # ***************
    # El vector de características empleado en el entrenamiento del Random-Forest será:
    # [Mín, Máx, Mediana, Percentil 25,Percentil 75] para Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Acc, Gyr.
    # self.X_train = data.X_train
    minimos_train = np.quantile(data, 0, axis=2, keepdims=True)
    maximos_train = np.quantile(data, 1, axis=2, keepdims=True)
    medianas_train = np.quantile(data, 0.5, axis=2, keepdims=True)
    Percentil_25_train = np.quantile(data, 0.25, axis=2, keepdims=True)
    Percentil_75_train = np.quantile(data, 0.75, axis=2, keepdims=True)
    Matriz_de_cuantiles_train = np.hstack((minimos_train, maximos_train, medianas_train, Percentil_25_train, Percentil_75_train))
    Matriz_de_cuantiles_train = np.squeeze(Matriz_de_cuantiles_train, axis=2)
    
    
    # *********************************
    # 2.- Características espectrales *
    # *********************************
    # Inicializamos las matrices de resultados
    num_filas = (data).shape[0]  # m ejemplos
    num_columnas = (data).shape[1]  # 12
    
    matriz_resultados_armonicos = np.zeros((num_filas,30))    # 1 IMU
    # matriz_resultados_armonicos = np.zeros((num_filas,60))    # 2 IMUs
    # Recorremos cada serie temporal y calculamos las características
    for i in range(num_filas):
        armonicos_totales = np.zeros((6,5))      # 1 IMU  
        # armonicos_totales = np.zeros((12,5))   # 2 IMUs
        for j in range(num_columnas):
            # Extraemos la serie temporal de longitud 250
            serie = data[i, j, :]
            # Calculamos las características espectrales
            resultado_armonicos,_ = obtener_caracteristicas_espectrales(serie,25)
            armonicos_totales[j, :] = resultado_armonicos
        armonicos_totales_2 = np.reshape(armonicos_totales,(1,-1))
        matriz_resultados_armonicos[i,:] = armonicos_totales_2
    
    
    # *****************************************
    # 3.- Número de picos y prominencia media *
    # *****************************************
    matriz_resultados_numero_picos = np.zeros((num_filas,12))   # 1 IMUs
    # matriz_resultados_numero_picos = np.zeros((num_filas,24))   # 2 IMUs
    # # Recorremos cada serie temporal y calculamos los picos
    for i in range(num_filas):  
        picos_totales = np.zeros(6)         # 1 IMU
        prominencias_totales = np.zeros(6)  # 1 IMUs
        # picos_totales = np.zeros(12)      # 2 IMUs
        # prominencias_totales = np.zeros(12) # 2 IMUs
        for j in range(num_columnas):
            # Extraemos la serie temporal de longitud 250
            serie = data[i, j, :]
            # Calculamos las características espectrales
            indices_picos, propiedades_picos = find_peaks(serie, prominence=True)
            numero_picos=len(indices_picos)
            if numero_picos > 0:
                # Si se detectaron picos, podemos proceder con el cálculo
                prominencias_picos = propiedades_picos['prominences']
                # Por ejemplo, calcular la mediana de la prominencia de los picos
                prominencia_media = np.median(prominencias_picos)
                #print(f"Mediana de prominencia: {prominencia_media}")
            else:
                # prominencia_media = np.NaN
                prominencia_media = 0
            
            # Guardamos los resultados en las matrices correspondientes
            picos_totales[j] = numero_picos
            prominencias_totales[j] = prominencia_media
            
        picos_totales_2 = np.reshape(picos_totales,(1,-1))
        prominencias_totales_2 = np.reshape(prominencias_totales,(1,-1))
        matriz_resultados_numero_picos[i,:] = np.hstack((picos_totales_2, prominencias_totales_2))
    
    
    # *******************
    # 4.- Correlaciones *
    # *******************
    matriz_correlaciones = np.zeros((num_filas,15))  # 1 IMU
    # matriz_correlaciones = np.zeros((num_filas,66))  # 2 IMUs
    for i in range(num_filas):
        # Calcular la matriz de correlación entre las filas
        correlacion = np.corrcoef(data[i,:,:], rowvar=True)
        # Extraer la parte superior de la matriz sin la diagonal principal
        upper_triangle_values = correlacion[np.triu_indices_from(correlacion, k=1)]
        # print(upper_triangle_values)
        
        matriz_correlaciones[i,:] = upper_triangle_values
    #self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones))
    #print(self.X_train)
    
    # **************************************
    # 5.- Autocorrelación del acelerómetro *
    # **************************************
    matriz_resultados_autocorrelacion = np.zeros((num_filas, 1))
    # matriz_resultados_autocorrelacion = np.zeros((num_filas, 2))
    # Recorremos cada serie temporal y calculamos los picos
    for i in range(num_filas):
        serie = np.linalg.norm(data[i,0:3,:], axis=0)
        # serie_desplazada = np.pad(serie[-25], (25,), mode='constant', constant_values=0)
        serie_desplazada = np.empty_like(serie)
        serie_desplazada[:25] = 0
        serie_desplazada[25:] = serie[:-25]
            
        autocorrelacion_acc_IMU1 = np.corrcoef(serie, serie_desplazada)

        serie = np.linalg.norm(data[i,6:9,:], axis=0)
        serie_desplazada = np.empty_like(serie)
        serie_desplazada[:25] = 0
        serie_desplazada[25:] = serie[:-25]
        # serie_desplazada = np.pad(serie[:,-25], (25,0), mode='constant', constant_values=0)
        autocorrelacion_acc_IMU2 = np.corrcoef(serie, serie_desplazada)
        
        # modulo_acc_IMU1 = np.linalg.norm(data.X_train[i,0:3,:], axis=0)
        # modulo_acc_IMU2 = np.linalg.norm(data.X_train[i,6:9,:], axis=0)
        # autocorrelacion_acc_IMU2 = np.corrcoef(modulo_acc_IMU2, nlags=25)
        
        matriz_resultados_autocorrelacion[i,0] = autocorrelacion_acc_IMU1[0,1]
        # matriz_resultados_autocorrelacion[i,1] = autocorrelacion_acc_IMU2[0,1]
    
    # self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion))      
    
    # **************************************************
    # 6.- Componentes roll, pitch y yaw del movimiento *
    # **************************************************
    dt = 1/25      # Período de muestreo en [s]
    rolls_promedio = np.zeros((num_filas, 1))
    pitches_promedio = np.zeros((num_filas, 1))
    yaws_promedio = np.zeros((num_filas, 1))
    for i in range(num_filas):
        rolls = []
        pitches = []
        yaws = []
        # Extraemos las series temporales de longitud 250 muestras (acelerómetro y giroscopio)
        serie_acc_x = data[i, 0, :]
        serie_acc_y = data[i, 1, :]
        serie_acc_z = data[i, 2, :]
        serie_gyr_x = data[i, 3, :]
        serie_gyr_y = data[i, 4, :]
        serie_gyr_z = data[i, 5, :]
        
        yaw_acumulado = 0
        for j in range(len(serie_acc_x)):
            acc_x = serie_acc_x[j]
            acc_y = serie_acc_y[j]
            acc_z = serie_acc_z[j]
            gyr_x = serie_gyr_x[j]
            gyr_y = serie_gyr_y[j]
            gyr_z = serie_gyr_z[j]

            roll = atan2(acc_y, acc_z)                             # Roll: rotación alrededor del eje X
            pitch = atan2(-acc_x, sqrt(acc_y**2 + acc_z**2))  # Pitch: rotación alrededor del eje Y
            yaw = gyr_z * dt                                            # Integración simple para obtener el cambio de yaw
            yaw_acumulado += yaw                                        # Efecto acumulativo de la acción integral
            rolls.append(roll)
            pitches.append(pitch)
        yaws.append(yaw_acumulado)
        yaw_acumulado = 0
        
        rolls_promedio[i] = np.mean(rolls)
        pitches_promedio[i] = np.mean(pitches)
        yaws_promedio[i] = np.mean(yaws)
    
    X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion, rolls_promedio, pitches_promedio, yaws_promedio))    

    return X_train

def extract_features_from_csv(csv_file, registro_actividades, body_segment='Thigh', 
                            window_size=250, sample_init=None):
    """
    Pipeline completo: carga de un CSV, enventanado y extracción de features para todas las actividades.

    Parameters:
    -----------
    csv_file : str
        Ruta al archivo CSV con datos de MATRIX
    registro_actividades : str
        Ruta al archivo Excel con el registro de actividades
    body_segment : str, optional
        Segmento corporal donde está colocado el IMU ('Thigh', 'Wrist', 'Hip')
        Default: 'Thigh'
    window_size : int, optional
        Tamaño de la ventana en muestras
        Default: 250
    sample_init : int, optional
        Muestra inicial para calibración (si se especifica)
        Default: None

    Returns:
    --------
    dict
        Diccionario con las siguientes claves:
        - 'features_by_activity': dict con features por actividad
        - 'windowed_data_by_activity': dict con datos enventanados por actividad
        - 'all_activities': lista de todas las actividades disponibles

    Example:
    --------
    >>> result = extract_features_from_csv(
    ...     './data/PMP1051_W1_PI.CSV',
    ...     './data/PMP1051_RegistroActividades.xlsx',
    ...     body_segment='Thigh',
    ...     window_size=250
    ... )
    >>> print(f"Actividades: {result['all_activities']}")
    >>> print(f"Features shape (actividad X): {result['features_by_activity']['X'].shape}")
    """
    from . import file_management, segmentation

    # 1. Cargar y escalar los datos desde el CSV
    scaled_data, timing_dict = file_management.load_scale_WPM_data(
        csv_file, body_segment, registro_actividades, sample_init
    )

    # 2. Segmentar los datos por actividad
    segmented_data = segmentation.segment_WPM_activity_data(timing_dict, scaled_data)

    available_activities = list(segmented_data.keys())
    features_by_activity = {}
    windowed_data_by_activity = {}

    # 3. Procesar todas las actividades
    for actividad in available_activities:
        data_actividad = segmented_data[actividad]
        # 4. Enventanar los datos
        windowed_dict = segmentation.apply_windowing_WPM_segmented_data({actividad: data_actividad}, window_size)
        windowed = windowed_dict[actividad]
        # Seleccionamos solo las columnas de interés (1:7 para Acc y Gyr)
        windowed_selected = windowed[:, 1:7, :]  # (num_ventanas, 6, window_size)
        # 5. Extraer features para cada ventana
        features = extract_features(windowed_selected, n_imus=1)
        features_by_activity[actividad] = features
        windowed_data_by_activity[actividad] = windowed_selected

    return {
        'features_by_activity': features_by_activity,
        'windowed_data_by_activity': windowed_data_by_activity,
        'all_activities': available_activities
    }

def extract_features_from_stack(stack_file):
    """
    Pipeline para extraer features desde un archivo NPZ que contiene un stack de datos enventanados.
    
    Parameters:
    -----------
    stack_file : str
        Ruta al archivo NPZ que contiene el stack de datos enventanados y sus etiquetas.
        El archivo debe contener:
        - 'concatenated_data': array con forma (num_ventanas, canales, window_size)
        - 'labels': array con las etiquetas correspondientes a cada ventana
    n_imus : int, optional
        Número de IMUs en el stack de datos
        Default: 2 (muslo y muñeca)
        
    Returns:
    --------
    dict
        Diccionario con las siguientes claves:
        - 'features': features extraídos con forma (num_ventanas, num_features)
        - 'labels': etiquetas correspondientes a cada ventana
        - 'windowed_data': datos enventanados originales
        - 'data_shape': forma de los datos originales
        - 'num_windows': número total de ventanas
        - 'unique_labels': etiquetas únicas en el dataset
    
    Example:
    --------
    >>> result = extract_features_from_stack(
    ...     './data/stacks/data_tot_PMP1020_1051.npz',
    ...     n_imus=2
    ... )
    >>> print(f"Features shape: {result['features'].shape}")
    >>> print(f"Unique labels: {result['unique_labels']}")
    """
    import numpy as np
    
    # 1. Cargar el stack y las etiquetas
    try:
        with np.load(stack_file, allow_pickle=True) as data:
            concatenated_data = data[WINDOW_CONCATENATED_DATA]  # (num_ventanas, canales, window_size)
            labels = data[WINDOW_ALL_LABELS]
            metadata = data[WINDOW_ALL_METADATA]
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo: {stack_file}")
    except KeyError as e:
        raise KeyError(f"El archivo NPZ no contiene la clave requerida: {e}. "
                      f"El archivo debe contener 'concatenated_data' y 'labels'.")
    
    # Validar las dimensiones de los datos
    if len(concatenated_data.shape) != 3:
        raise ValueError(f"Los datos deben tener 3 dimensiones (num_ventanas, canales, window_size), "
                        f"pero tienen forma: {concatenated_data.shape}")
    
    if len(concatenated_data) != len(labels):
        raise ValueError(f"El número de ventanas ({len(concatenated_data)}) no coincide "
                        f"con el número de etiquetas ({len(labels)})")
    
    # 2. Extraer features para cada ventana del stack
    features = extract_features(concatenated_data)
    
    # 3. Obtener información adicional
    # unique_labels = np.unique(labels)
    
    # return {
    #     'features': features,
    #     'labels': labels,
    #     'windowed_data': concatenated_data,
    #     'data_shape': concatenated_data.shape,
    #     'num_windows': len(concatenated_data),
    #     'unique_labels': unique_labels
    # }

    return features, labels, metadata