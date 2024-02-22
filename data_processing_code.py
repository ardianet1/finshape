# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:59:39 2023

@author: Hadrien Sprumont
"""

import numpy as np
import pandas as pd
import re

import atiiaftt

from pathlib import Path
from dataclasses import dataclass, field

import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'Warning: {msg}\n' # Monkey patching

from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter as medfilt


# ========== Filter algorithms ========== #
def norm_data_shape(data):
    if len(data.shape) == 1:
        return data.reshape(1,-1)
    else:
        return data.copy()

def low_pass_filter(data, alpha):
    data = norm_data_shape(data)
    
    filtered_data = np.copy(data)
    for i in range(filtered_data.shape[0]):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    return filtered_data

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order,normal_cutoff, btype='low',analog=False)
    return b,a

def butter_lowpass_filter(data, cutoff=0.06, fs=1.0, order=5):
    data = norm_data_shape(data)
    
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = filtfilt(b, a, data[i])
    return filtered_data
    
def median_filter(data, window_size):
    data = norm_data_shape(data)
    
    filtered_data = np.copy(data)
    for i in range(data.shape[0]):
        filtered_data[i] = medfilt(data[i], size=window_size)
    return filtered_data


# ========== Custom Errors and Warnings ========== #
class SaturationError(Exception): pass

class SaturationWarning(UserWarning): pass


# ========== Measurement Class ========== #
@dataclass
class Meas():
    
    meas_id: int
        
    ft_is_raw_voltage: bool = True
        
    # ===== Files ===== #
    ref_file_path: Path = Path("../Flow Tank Experiment List 22_11_2023.xlsx").resolve()
    ft_calibration_file: Path = Path("../FT41361.cal").resolve()
    ft_file_path: Path = None
    rio_file_path: Path = None
        
    ft_folder: str = "FT_data_23.11.2023"
    id_suffix: str = ""
        
    # ===== Identifiers ===== #
    fin_type: str = None
    freq: float = None
    duty: float = None
    flow: float = None
        
    # ===== Data and processed values ===== #
    raw_voltages: np.ndarray = None # With shape (6,X)
    raw_data: np.ndarray = None # With shape (6,X) ! Raw data is not unbiased
    data: np.ndarray = None # With shape (6,X)
    ft_timestamps: np.array = None # With shape (X)
        
    front_bias: np.ndarray = None # With shape (6)
    back_bias: np.ndarray = None # With shape (6)
        
    mean: np.ndarray = None # With shape (6)
    std: np.ndarray = None # With shape (6)
        
    # Unfiltered and filtered periods without saturation (key = index w/ respect to raw_data index)
    periods_raw: dict[int:np.ndarray] = None # ! Raw data is not unbiased
    periods: dict[int:np.ndarray] = None
    nb_errors: int = None
        
    # ===== Processing parameters ===== #
    min_nb_good_periods: int = 5
    max_nb_periods: int = 10
    
    sample_per_s: float = None
    sample_per_period: int = None
        
    # ===== Methods ===== #
    def __repr__(self): return str(self)
    
    def __str__(self): return f"Meas: {self.meas_id}"
    
    def __post_init__(self):
        
        # ===== Get the run data from the file ===== #
        self.get_meas_info()
        
        # ===== Process the raw data ===== #
        output = self.process_raw_data()
        
        # Storing all the data
        self.data = output['data']
        self.front_bias = output['front_bias']
        self.back_bias = output['back_bias']
        self.periods_raw = output['periods_raw']
        self.periods = output['periods']
        self.nb_errors = output['nb_errors']
        self.mean = output['mean']
        self.std = output['std']
        
        # ===== Test data validity ===== #
        if len(self.periods) < self.min_nb_good_periods:
            raise SaturationError(f"Not enough usable periods for measurement:'{self.run_id}'")
        
    def get_meas_info(self) -> dict:
        
        # ===== Extract identifiers from the reference file ===== #
        df = pd.read_excel(self.ref_file_path, sheet_name="Open Loop")
        run_name = f"OL_{self.meas_id}" if self.meas_id > 9 else f"OL_0{self.meas_id}"
        df = df[df["Run Name"] == run_name]
        
        ft_file_path = Path(self.ref_file_path.parent,self.ft_folder,f"{run_name}{self.id_suffix}.csv")
        rio_file_path = Path(self.ref_file_path.parent,"myRIO_data_22.11.2023",f"{run_name}.csv")
        
        fin_type = df["Fin Type"].item()
        frequency = df["Frequency [Hz]"].item()
        duty_cycle = df["Duty Cycle"].item()
        flow_speed = df["Calc. Water Speed [cm/s]"].item()
        
        # ===== Extract the raw data from the csv file ===== #
        df = pd.read_csv(ft_file_path,header=None)
        ft_timestamps = df[df.columns[-1]].to_numpy() # Store timestamps
        df.drop(df.columns[-1],axis=1,inplace=True) # Remove timestamps
        data = df.to_numpy().T # Transpose the data to have the right shape (6,X)
        
        # ===== Store Values ===== #
        self.fin_type = fin_type
        self.freq = frequency
        self.duty = duty_cycle
        self.flow = flow_speed
        
        self.sample_per_s = 1000 / 1 # 1000 Hz, no averaging
        self.sample_per_period = int(self.sample_per_s / self.freq)
        
        self.ft_file_path = ft_file_path
        self.rio_file_path = rio_file_path
        
        self.ft_timestamps = ft_timestamps
        if self.ft_is_raw_voltage:
            self.raw_voltages = data
            self.raw_data = self.ft_convert(data)
        else:
            self.raw_data = data
        
        return fin_type, frequency, duty_cycle, flow_speed, data

    def ft_convert(self, voltages):
        #TODO Cleanup
        
        v = voltages.copy()
        
        tool_transform = [0,0,0,90,-135,-90]

        ft_converter = atiiaftt.FTSensor(self.ft_calibration_file.as_posix(), 1)
        ft_converter.setToolTransform(tool_transform, atiiaftt.FTUnit.DIST_MM, atiiaftt.FTUnit.ANGLE_DEG)
        ft_converter.setForceUnits(atiiaftt.FTUnit.FORCE_N)
        ft_converter.setTorqueUnits(atiiaftt.FTUnit.TORQUE_N_MM)
        
        ft_values = np.zeros(v.shape).T
        for i in range(v.shape[1]):
            values = v[:,i]
            if any(np.abs(values) > 0.99*10):
                values = np.zeros(values.shape)
            ft_values[i,:] = ft_converter.convertToFt(values.tolist())
        
        return ft_values.T
        
    def process_raw_data(self) -> dict:
        
        data = self.raw_data.copy()
        
        # Remove idle sections at the beginning and at the end
        data, front_bias, front_index = self.front_processing(data)
        data, back_bias, back_index = self.back_processing(data)
        
        # Get rid of the saturated points
        # (! Bias should not be removed before this call, otherwise saturation cannot be detected !)
        valid_periods, nb_errors = self.period_processing(data)
        
        # Filter the periods and store the data
        concat_data = np.zeros((self.raw_data.shape[0],0))
        period_means = []
        filtered_periods = {}
        for key in valid_periods:
            
            period = valid_periods[key].copy()
            
            # Filtering via median + butterworth filters
            period = median_filter(period, window_size=5*16)
            # period = butter_lowpass_filter(period, cutoff=self.freq*0.06, fs=self.freq, order=2)
            period = butter_lowpass_filter(period, cutoff=60, fs=1000, order=2)
            
            # Remove bias
            period -= back_bias.reshape((-1,1))
            
            filtered_periods[key] = period
            concat_data = np.concatenate((concat_data, period), axis=1)
            period_means.append(period.mean(axis=1))
            
        # Mean and std of the run
        nb_periods = len(period_means)
        if nb_periods > 0:
            mean = np.mean(period_means[-min(nb_periods, self.max_nb_periods):], axis=0)
            std = np.std(period_means[-min(nb_periods, self.max_nb_periods):], axis=0)
        else:
            mean = None
            std = None
            
        # Storing all the data
        output = {}
        output['data'] = concat_data
        output['front_bias'] = front_bias
        output['back_bias'] = back_bias
        output['periods_raw'] = valid_periods
        output['periods'] = filtered_periods
        output['nb_errors'] = nb_errors
        output['mean'] = mean
        output['std'] = std
            
        return output
           
    def front_processing(self, raw_data) -> (np.ndarray, np.ndarray, int):
        # Removes the extra samples at the beginning, where the fish is inactive
        # The front bias is computed as the average value of those extra samples
        # Start of activity is detected when a Z-torque value differs too much from the front bias
        # (Too much = more than 20% of the maximum Z-torque)
        
        # We test the Z-torque as it has the best SNR
        z_torque_index = 5 
        test_data = raw_data[z_torque_index].copy()
        test_data = np.abs(test_data-test_data.mean()) # Center the test data and abs
        
        max_value = test_data.max()
        
        index = 1 # Index of the start of activity
        while (index < test_data.shape[0]) and (abs(test_data[:index].mean() - test_data[index]) < 0.2*max_value):
            index += 1
        
        if index < 100: # Not idle enough -> In motion from the start
            index = 0
            data = raw_data.copy()
            front_bias = np.zeros(raw_data.shape[0])
            
        elif index > test_data.shape[0]-100: # Idle for too long -> No activity detected
            raise ValueError("Unable to detect start of activity")
            
        else: # We can remove the idle section and compute the bias
            index += self.sample_per_period
            data = raw_data[:,index:].copy()
            front_bias = raw_data[:,:index].mean(1)
                        
        return data, front_bias, index
    
    def back_processing(self, raw_data) -> (np.ndarray, np.ndarray, int):
        # Removes the extra samples at the end, where the fish is inactive
        # The back bias is computed as the average value of those extra samples
        # Uses the previously defined 'front_processing' with flipped data
        
        flipped_data, back_bias, flipped_index = self.front_processing(np.flip(raw_data, axis=1))
        
        index = raw_data.shape[1] - flipped_index
        data = np.flip(flipped_data, axis=1)
        
        return data, back_bias, index
    
    def period_processing(self, raw_data) -> (dict[int:np.ndarray], int):
        # Isolates as many periods as possible that do not possess saturated data
        
        data = raw_data.copy()
        
        len_period = self.sample_per_period
        valid_periods = {} # To store the periods, the starting index is the key
        nb_errors = 0 # Count the number of saturated points
        
        index = 0
        while index < raw_data.shape[1]-len_period:
            
            subset = data[:,index:index+len_period]
            
            if np.sum(subset == 0) < 6: # No saturation in the subset
                valid_periods[index] = subset
                index += len_period
            else:
                nb_errors += 1
                index += 1
        
        return valid_periods, nb_errors

    @property
    def df(self):
        
        return pd.DataFrame(self.raw_data.T,None,
                            ["Forward Force (X) [N]","Side Force (Y) [N]","Vertical Force (Z) [N]",
                             "Roll (rX) [Nmm]","Pitch (rY) [Nmm]","Yaw (rZ) [Nmm]"],copy=True)
    
# ========== Run Class ========== #
@dataclass
class Run():
    """A Run is a collection of Measurements"""
    
    # To find the measurement files
    meas_ids: list[int] = field(default_factory=list)
    folder: str = "FT_data_23.11.2023"
    suffixes: list[str] = None
    
    # Run identifier
    run_name: str = "unnamed"
    
    # Extra parameters
    max_nb_periods: int = 10
    
    # Data
    measurements: list[Meas] = field(default_factory=list)
    nb_errors: int = None
    bias: list[float] = field(default_factory=lambda:[0,0,0,0,0,0])
    
    warn: bool = True
    
    # ===== Methods ===== #
    def __repr__(self): return str(self)
    
    def __str__(self): return f"Run: {self.run_name} -> {self.meas_ids}"
    
    def __post_init__(self):
        
        # ===== Load all measurements ===== #
        self.measurements, self.nb_errors = self.get_measurements()
        
    def __add__(self, other):
        
        if not isinstance(other, Run):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        
        new = Run()
        new.measurements = self.measurements + other.measurements
        new.meas_ids = self.meas_ids + other.meas_ids
        new.warn = self.warn or other.warn
        new.run_name = " + ".join([c for c in np.unique(self.run_name.split(" + ")+other.run_name.split(" + "))])
        
        return new
        
    def get_measurements(self) -> (list[Meas], int):
        
        if self.suffixes is None:
            suffix_list = ["" for i in self.meas_ids]
        else:
            suffix_list = self.suffixes
            
        measurements = []
        nb_errors = 0
        for i,m_id in enumerate(self.meas_ids):
            try: 
                measurements.append(
                    Meas(
                        m_id,
                        ft_folder=self.folder,
                        id_suffix=suffix_list[i],
                        max_nb_periods=self.max_nb_periods
                        )
                )
            except SaturationError as e: 
                nb_errors += 1
                if self.warn: warnings.warn(str(e), SaturationWarning)
            
        if self.warn and nb_errors != 0:
            warnings.warn(f"Processing revealed {nb_errors} / {len(measurements)} saturated measurements in run '{self.run_name}'", SaturationWarning)
            
        return measurements, nb_errors
    
    def get_line(self, axis:str, index=0, fins=None, freqs=None, duty_cycles=None, flows=None) -> (np.array, np.array, np.array):
        # Returns a tuple with information to plot a line of results.
        # You can restrict the set using 'freqs', 'duty_cycles', and 'flows'
        
        if fins is None: fins = self.fins
        elif not isinstance(fins, list): fins = [fins]
        if freqs is None: freqs = self.freqs
        elif not isinstance(freqs, list): freqs = [freqs]
        if duty_cycles is None: duty_cycles = self.duty_cycles
        elif not isinstance(duty_cycles, list): duty_cycles = [duty_cycles]
        if flows is None: flows = self.flows  
        elif not isinstance(flows, list): flows = [flows]
        
        x, y, y_err = [], [], []
        for m in self.measurements:
            pt = {"freq":m.freq, "duty":m.duty, "flow":m.flow}
            if m.fin_type not in fins: continue
            if m.freq not in freqs: continue
            if m.duty not in duty_cycles: continue
            if m.flow not in flows: continue
            x.append(pt[axis])
            y.append(m.mean[index] - self.bias[index])
            y_err.append(m.std[index])
        
        x, y, y_err = map(list,zip(*sorted(zip(x,y,y_err),key=lambda a: a[0])))
        
        return np.array(x), np.array(y), np.array(y_err)

    # ===== Properties ===== #
    @property
    def fins(self) -> list[str]: 
        return list(sorted(np.unique([m.fin_type for m in self.measurements]), key=
        lambda x: ["Truncated","Carthorhyncus","Utatsusaurus",
                   "Mixosaurus","Guizhouichthyosaurus","Ophthalmosaurus"].index(x)))
    
    @property
    def freqs(self) -> list[float]: return list(np.unique(sorted([m.freq for m in self.measurements])))
    
    @property
    def duty_cycles(self) -> list[float]: return list(sorted(np.unique([m.duty for m in self.measurements])))
    
    @property
    def flows(self) -> list[float]: return list(sorted(np.unique([m.flow for m in self.measurements])))
    
    @property
    def results(self) -> dict[(float,float,float):np.array]: 
        
        results = {}
        for m in self.measurements:
            results[(m.freq, m.duty, m.flow)] = m.mean
        
        return results
    
def buo(duty_cycle):
    
    df = pd.read_csv("../right_filled.csv")
    single = df[df.columns[:6]].to_numpy().mean(0)
    df = pd.read_csv("../both_filled.csv")
    both = df[df.columns[:6]].to_numpy().mean(0)
    
    bias = both*(duty_cycle-0.5) + single*(1-duty_cycle+0.5)
    
    bias[1] = 0
    bias[3] = 0
    bias[5] = 0
    
    return bias

def get_flow_linear_regression(run:Run):
    
    pts = []
    for m in run.measurements:
        for p in list(m.periods.values())[-10:]:
            pts.append([m.flow, p.mean(1)[0]])
    pts = np.array(pts).T.copy()
    pts[1] = pts[1]-buo(m.duty)[0]
    
    results = scipy.stats.linregress(pts[0],pts[1])
    
    return pts, results