import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
import math
import pywt
from kneed import KneeLocator
from scipy.signal import find_peaks

class DegradationAnalyzer:
    """
    Analyzes battery degradation by tracking features from ICA and DTV curves.
    """    
    def __init__(self, battery_data):
        """
        Initializes the analyzer with processed battery data.

        Args:
            battery_data (dict): The dictionary returned by DataInitializer.load_Experimental.
        """
        self.data = battery_data
        self.window_size_f = 20
        self.SG_order = 5
        print("DegradationAnalyzer initialized.")

    def filter_SG(self, x):
        if len(x) >= self.window_size_f:
            return savgol_filter(x, self.window_size_f, self.SG_order)
        else:
            return x
        
    def find_features(self, feature_type='ICA', charge_discharge='Charge', find='peaks', 
                      voltage_windows=None, peak_params=None):
        """
        Finds peaks or valleys in ICA or DTV curves within specified voltage windows.

        Args:
            feature_type (str): The feature to analyze, either 'ICA' or 'DTV'.
            charge_discharge (str): The cycle type, either 'charge' or 'dch'.
            find (str): The feature to find, either 'peaks' or 'valleys'.
            voltage_windows (dict): A dictionary defining the voltage range for each peak. 
                                    Example: {'P1': (3.6, 3.8), 'P2': (3.9, 4.1)}
            peak_params (dict): Parameters for scipy.signal.find_peaks (e.g., prominence, width).

        Returns:
            dict: A dictionary containing the evolution of each feature (voltage, magnitude, cycle, capacity).
        """
        if voltage_windows is None:
            print("Error: Please provide 'voltage_windows' to track features.")
            return None
        if peak_params is None:
            peak_params = {'prominence': 0.01} # Default prominence

        # Determine which data to use
        mode_key = 'Original_ch' if charge_discharge == 'Charge' else 'Original_dch'
        cap_key = 'ch_cap' if charge_discharge == 'Charge' else 'dch_cap'

        # Initialize results structure
        results = {
            name: {'voltage': [], 'magnitude': [], 'cycle': [], 'capacity': []}
            for name in voltage_windows
        }
        
        # Unpack data for iteration
        cycles = self.data['cycles']
        capacities = self.data[cap_key]
        signals = self.data[mode_key][feature_type]
        voltages = self.data[mode_key]['Voltage']

        print(f"Analyzing {len(cycles)} cycles for {feature_type} {find} during {charge_discharge}...")

        # Loop through each cycle
        for i, cycle in enumerate(cycles):
            signal_y = np.array(signals[i])
            voltage_x = np.array(voltages[i])

            if len(signal_y) < 3 or len(voltage_x) < 3:
                for name in voltage_windows:
                    results[name]['voltage'].append(np.nan)
                    results[name]['magnitude'].append(np.nan)
                    results[name]['cycle'].append(cycle)
                    results[name]['capacity'].append(capacities[i])
                continue

            # Invert signal to find valleys using find_peaks
            analysis_signal = -signal_y if find == 'valleys' else signal_y

            # Find all peaks in the current cycle's signal
            peak_indices, properties = find_peaks(analysis_signal, **peak_params)
            
            # For each defined window, find the best peak
            for name, (vmin, vmax) in voltage_windows.items():
                
                # Filter peaks that are within the current voltage window
                window_peak_indices = [
                    p_idx for p_idx in peak_indices 
                    if vmin <= voltage_x[p_idx] <= vmax
                ]

                if not window_peak_indices:
                    # No peak found in this window for this cycle
                    found_peak = False
                else:
                    # If multiple peaks are found, select the most prominent one
                    prominences = properties['prominences']
                    peak_prominences_in_window = [
                        prom for p_idx, prom in zip(peak_indices, prominences) 
                        if p_idx in window_peak_indices
                    ]
                    
                    # Get the index of the most prominent peak within the window
                    most_prominent_idx_local = np.argmax(peak_prominences_in_window)
                    final_peak_idx = window_peak_indices[most_prominent_idx_local]
                    
                    # Store its properties
                    results[name]['voltage'].append(voltage_x[final_peak_idx])
                    results[name]['magnitude'].append(signal_y[final_peak_idx])
                    found_peak = True

                # If no peak was found, append NaN
                if not found_peak:
                    results[name]['voltage'].append(np.nan)
                    results[name]['magnitude'].append(np.nan)

                # Always append cycle and capacity info
                results[name]['cycle'].append(cycle)
                results[name]['capacity'].append(capacities[i])

        return results

    def plot_feature_evolution(self, feature_results, mode, feature_name='Feature', label = 'ICA', cycle_start = None, cycle_end = None):
        """
        Plots the evolution of extracted peak/valley features against cycle and capacity.

        Args:
            feature_results (dict): The dictionary returned by find_features.
            feature_name (str): The name of the feature for plot titles (e.g., "ICA Charge Peaks").
        """
        if cycle_start == None:
            cycle_start = min(self.data['cycles'])
        
        if cycle_end == None:
            cycle_end = max(self.data['cycles'])

        if feature_results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Evolution of {feature_name}', fontsize=20)
        
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']

        fs_label = 16       # Axis labels
        fs_ticks = 14       # Axis tick numbers
        fs_title = 18       # Plot title
        fs_legend = 13      # Legend text

        num_items = len(feature_results)
        color_range = plt.cm.viridis(np.linspace(0.0, 0.7, num_items))

        for i, (name, values) in enumerate(feature_results.items()):
            marker = markers[i % len(markers)]
            color = color_range[i]
            
            # Unpack data, converting to numpy arrays for easier plotting
            cycles = np.array(values['cycle'])
            idx = np.where((cycles >= cycle_start) & (cycles <= cycle_end))
            cycles = cycles[idx]
            capacities = np.array(values['capacity'])[idx]
            magnitudes = np.array(values['magnitude'])[idx]
            voltages = np.array(values['voltage'])[idx]

            # Plot 1: Magnitude vs. Cycle Index
            ax = axes[0, 0]
            ax.plot(cycles, magnitudes, marker=marker, linestyle='-', label=name, color=color)
            ax.set_title(label+' Amplitude vs. Cycle', fontsize=fs_title)
            ax.set_xlabel('Cycle Index', fontsize=fs_label)
            ax.set_ylabel(mode+' Amplitude', fontsize=fs_label)
            ax.tick_params(axis='x', labelsize=fs_ticks)
            ax.tick_params(axis='y', labelsize=fs_ticks)     
            ax.grid(True)
            ax.legend(prop={'size': fs_legend}, fontsize=fs_legend)

            # Plot 2: Voltage vs. Cycle Index
            ax = axes[0, 1]
            ax.plot(cycles, voltages, marker=marker, linestyle='-', label=name, color=color)
            ax.set_title('Voltage vs. Cycle', fontsize=fs_title)
            ax.set_xlabel('Cycle Index', fontsize=fs_label)
            ax.set_ylabel(mode+'Voltage (V)', fontsize=fs_label)  
            ax.tick_params(axis='x', labelsize=fs_ticks)
            ax.tick_params(axis='y', labelsize=fs_ticks) 
            ax.grid(True)
            ax.legend(prop={'size': fs_legend}, fontsize=fs_legend)
            
            # Plot 3: Magnitude vs. Capacity
            ax = axes[1, 0]
            ax.plot(capacities, magnitudes, marker=marker, linestyle='-', label=name, color=color)
            ax.set_title(label+' Amplitude vs. Capacity', fontsize=fs_title)
            ax.set_xlabel('Capacity (Ah)', fontsize=fs_label)
            ax.set_ylabel(mode+' Amplitude', fontsize=fs_label)
            ax.tick_params(axis='x', labelsize=fs_ticks)
            ax.tick_params(axis='y', labelsize=fs_ticks)   
            ax.grid(True)
            ax.legend(prop={'size': fs_legend}, fontsize=fs_legend)

            # Plot 4: Voltage vs. Capacity
            ax = axes[1, 1]
            ax.plot(capacities, voltages, marker=marker, linestyle='-', label=name, color=color)
            ax.set_title('Voltage vs. Capacity', fontsize=fs_title)
            ax.set_xlabel('Capacity (Ah)', fontsize=fs_label)
            ax.set_ylabel(mode+'Voltage (V)', fontsize=fs_label)
            ax.tick_params(axis='x', labelsize=fs_ticks)
            ax.tick_params(axis='y', labelsize=fs_ticks)    
            ax.grid(True)
            ax.legend(prop={'size': fs_legend}, fontsize=fs_legend)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

class DataInitializer:
    def __init__(self, cutoff=0.1, fs=2, order=2, window_size_f1=30, window_size_f2=30, SG_order1 = 5, SG_order2 = 5, C_init= None):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.window_size_f1 = window_size_f1
        self.window_size_f2 = window_size_f2
        self.window_size_f3 = 50
        self.SG_order1 = SG_order1
        self.SG_order2 = SG_order2
        self.SG_order3 = 5
        self.C_init = C_init
        


    def wavelet_denoising(self, data, wavelet='db4', level=2):
        data = np.array(data)
        coeff = pywt.wavedec(data, wavelet, mode="per")
        sigma = np.median(np.abs(coeff[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode='per')
        return reconstructed_signal[:len(data)]
    
    def low_pass_filter(self, data):
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    
    def filter_SG1(self, x):
        if len(x) >= self.window_size_f1:
            return savgol_filter(x, self.window_size_f1, self.SG_order1)
        else:
            return x

    def filter_SG2(self, x):
        if len(x) >= self.window_size_f2:
            return savgol_filter(x, self.window_size_f2, self.SG_order2)
        else:
            return x
        
    def filter_SG3(self, x):
        if len(x) >= self.window_size_f3:
            return savgol_filter(x, self.window_size_f3, self.SG_order3)
        else:
            return x
                
    def length_equalizer(self, input_list, len_eq = None):    
        # Handle case where the entire input list is empty
        if not input_list:
            return np.array([])
            
        # Safely find the max length, ignoring any empty signals in the list
        if len_eq:
            max_length = len_eq
        else:
            valid_lengths = [len(signal) for signal in input_list if hasattr(signal, '__len__') and len(signal) > 0]
            if not valid_lengths:
                return np.array([[] for _ in input_list]) # Return list of empty arrays if all were empty
            max_length = max(valid_lengths)

        uniformed_signals = []
        for signal in input_list:
            # Check if the signal is empty or not a sequence
            if not hasattr(signal, '__len__') or len(signal) == 0:
                # If empty, append an array of zeros as a placeholder
                uniformed_signals.append(np.zeros(max_length))
                continue

            # Proceed with interpolation for valid signals
            original_indices = np.linspace(0, len(signal) - 1, num=len(signal))
            target_indices = np.linspace(0, len(signal) - 1, num=max_length)
            interpolated_signal = np.interp(target_indices, original_indices, signal)
            uniformed_signals.append(interpolated_signal)
            
        return np.array(uniformed_signals)
    
    def load_Experimental(self, file_path, min_temp = None, max_temp = None):
        print("______________")
        """
        Loads and preprocesses battery cycling data from a file.

        This method reads the raw data, resamples it, calculates differential
        temperature/voltage (dT/dV), separates charge and discharge cycles,
        and extracts numerous health indicators (HIs).

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            dict: A dictionary containing the processed data, including:
                  - 'data_dch', 'data_ch': DataFrames for discharge/charge steps.
                  - 'Original_dch', 'Original_ch': Dictionaries of unequalized time-series data.
                  - 'Equalized_dch', 'Equalized_ch': Dictionaries of equalized time-series data.
                  - 'cycles': A list of cycle numbers.
                  - 'dch_cap', 'ch_cap': Lists of discharge/charge capacities per cycle.
                  - 'HIs_dch', 'HIs_ch': Dictionaries of health indicators for each cycle.
        """
        alldata = pd.read_csv(file_path, header=None)
        alldata.columns = ['DataPoint', 'Cycle', 'Step Type', 'Time','Total Time', 'Current', 'Voltage',
                           'Capacity', 'Temperature']
        
        if file_path[-9:] == 'V45-B.csv':
            cycles = sorted(alldata['Cycle'].unique())[0:-5]
            alldata = alldata[alldata['Cycle'].isin(cycles)]

        if file_path[-12:] == 'V42-B-LT.csv':
            cycles = sorted(alldata['Cycle'].unique())[2:-5]
            alldata = alldata[alldata['Cycle'].isin(cycles)]

        resample_interval = int(2)
        alldata = alldata.iloc[::resample_interval, :]
        alldata['Temperature'] = alldata.groupby('Step Type')['Temperature'].transform(self.low_pass_filter)
        alldata['Temperature'] = alldata.groupby('Step Type')['Temperature'].transform(self.filter_SG1)
        #alldata['Current'] = alldata.groupby('Step Type')['Current'].transform(self.filter_SG1)       
        #alldata['Voltage'] = alldata.groupby('Cycle')['Voltage'].transform(self.low_pass_filter)    
        # Calculate and filter dT/dV
        alldata['dT/dV'] = alldata.groupby('Step Type')['Temperature'].diff() / alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dT/dV'] = alldata.groupby('Step Type')['dT/dV'].transform(self.filter_SG2)
        alldata['-dT/dV'] = -1 * alldata['dT/dV']
        alldata['dT/dt'] = alldata.groupby('Step Type')['Temperature'].diff()
        alldata['dT/dt'] = alldata.groupby('Step Type')['dT/dt'].transform(self.filter_SG2)
        alldata['dV/dt'] = alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dV/dQ'] = alldata.groupby('Step Type')['Voltage'].diff()/alldata.groupby('Step Type')['Capacity'].diff()
        alldata['dQ/dV'] = alldata.groupby('Step Type')['Capacity'].diff()/alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dQ/dV'] = alldata.groupby('Step Type')['dQ/dV'].transform(self.filter_SG2)
        alldata['dQ/dV'] = alldata.groupby('Step Type')['dQ/dV'].transform(self.filter_SG1)
        alldata['Time'] = pd.to_timedelta(alldata['Time']).dt.total_seconds()

        # Handle infinite values and fill NaNs
        alldata = alldata.replace([np.inf, -np.inf], np.nan)
        alldata = alldata.bfill()

        # Separate charge and discharge data
        data_dch_all = alldata.loc[alldata['Step Type'] == 'CC DChg'].copy()
        # Filter discharge cycles with current < -0.5
        data_dch_filtered = data_dch_all.loc[data_dch_all['Current'] < -0.5].copy()

        # Identify cycles where the last time value is < 2900s
        valid_cycles = data_dch_filtered.groupby('Cycle')['Time'].max()
        time_limit = 2900 if file_path[-9:] == 'V42-C.csv' else 7600
        valid_cycles = valid_cycles[valid_cycles < time_limit].index

        # Apply both filters
        data_dch = data_dch_filtered[data_dch_filtered['Cycle'].isin(valid_cycles)].copy()
        data_ch = alldata.loc[alldata['Step Type'] == 'CC Chg'].copy()

        #cycles_to_remove = [49,99] if file_path[-10:] == 'V435-A.csv' else []

        if file_path[-12:] == 'V42-B-LT.csv' or file_path[-12:] == 'V42-B-HT.csv' or file_path[-7:] == 'OD1.csv' or file_path[-7:] == 'OD2.csv':
            cycles_to_remove = []
        elif file_path[-9:] == 'V42-A.csv':
            cycles_to_remove = [181,182,183,184,185]
        else:
            cycles_to_remove = [49,99,98,54,197,246] #[98,99, 197]


        if min_temp is not None and max_temp is not None:
            cycles_to_check = sorted(alldata['Cycle'].unique())[:-1]
            for cycle in cycles_to_check:
                temp_dch = data_dch.loc[data_dch['Cycle'] == cycle, 'Temperature']
                temp_ch = data_ch.loc[data_ch['Cycle'] == cycle, 'Temperature']
                if (not temp_dch.empty and (temp_dch.min() < min_temp or temp_dch.max() > max_temp)) or \
                (not temp_ch.empty and (temp_ch.min() < min_temp or temp_ch.max() > max_temp)):
                    cycles_to_remove.append(cycle)

        # Remove the identified cycles from the DataFrames
        filtered_data = alldata[~alldata['Cycle'].isin(cycles_to_remove)].copy()
        data_dch = data_dch[~data_dch['Cycle'].isin(cycles_to_remove)].copy()
        data_ch = data_ch[~data_ch['Cycle'].isin(cycles_to_remove)].copy()
        print(f"Removed {len(cycles_to_remove)} cycles due to temperature violations.")

        if file_path[-9:] == 'V45-B.csv':
            cycles = sorted(data_dch['Cycle'].unique())[:-5]
        elif file_path[-12:] == 'V42-B-HT.csv' or file_path[-7:] == 'OD1.csv' or file_path[-7:] == 'OD2.csv':
            cycles = sorted(data_dch['Cycle'].unique())
        else:
            cycles = sorted(data_dch['Cycle'].unique())[:-1]

        # --- Initialize data storage lists and dictionaries ---
        dch_cap, ch_cap = [], []

        # Time-series data
        time_dch, voltage_dch, current_dch, temperature_dch, DTV_dch, DT_dch, DV_dch, ICA_dch, C_dch = [], [], [], [], [], [], [], [], []
        time_ch, voltage_ch, current_ch, temperature_ch, DTV_ch, DT_ch, DV_ch, ICA_ch, C_ch = [], [], [], [], [], [], [], [], []

        # Health Indicator dictionaries
        HIs_dch = {
            'discharge_time': [], 'temperature_std': [], 'voltage_std': [],
            'dv_dt_mean': [], 'dv_dt_std': [], 'current_std': [],
            'mean_temp': [], 'max_temp': [], 'voltage_integral': [], 'dv_dt_min': [], 'temp_rise': [],
            'knee_voltage': [], 'cc_transition_voltage': [], 'knee_transition_diff': [], 'starting_voltage': []
        }
        HIs_ch = {
            'charge_time': [], 'voltage_plateau_time': [], 'voltage_at_80_soc': [],
            'temperature_std': [], 'voltage_std': [], 'dv_dt_mean': [], 'dv_dt_std': [],
            'current_std': [], 'mean_temp': [], 'max_temp': [], 'voltage_integral': [], 'temp_rise': [],
            'knee_voltage': [], 'cc_transition_voltage': [], 'knee_transition_diff': [], 'starting_voltage': []
        }

        # --- Loop through cycles to extract data and calculate HIs ---
        for cycle in cycles:
            if cycle not in []:
                cycle_data_dch = data_dch[data_dch['Cycle'] == cycle]
                cycle_data_ch = data_ch[data_ch['Cycle'] == cycle]

                # Append time-series data
                time_dch.append(cycle_data_dch['Time'].values)
                voltage_dch.append(cycle_data_dch['Voltage'].values)
                current_dch.append(cycle_data_dch['Current'].values)
                temperature_dch.append(cycle_data_dch['Temperature'].values)
                DTV_dch.append(cycle_data_dch['dT/dV'].values)
                DT_dch.append(cycle_data_dch['dT/dt'].values)
                DV_dch.append(cycle_data_dch['dV/dt'].values)
                ICA_dch.append(cycle_data_dch['dQ/dV'].values)
                C_dch.append(cycle_data_dch['Capacity'].values)

                time_ch.append(cycle_data_ch['Time'].values)
                voltage_ch.append(cycle_data_ch['Voltage'].values)
                current_ch.append(cycle_data_ch['Current'].values)
                temperature_ch.append(cycle_data_ch['Temperature'].values)
                DTV_ch.append(cycle_data_ch['dT/dV'].values)
                DT_ch.append(cycle_data_ch['dT/dt'].values)
                DV_ch.append(cycle_data_ch['dV/dt'].values)
                ICA_ch.append(cycle_data_ch['dQ/dV'].values)
                C_ch.append(cycle_data_ch['Capacity'].values)


                # Append capacity
                dch_cap.append(np.max(cycle_data_dch['Capacity'].dropna()))
                ch_cap.append(np.max(cycle_data_ch['Capacity'].dropna()))

                # --- Calculate and Store Health Indicators (HIs) ---

                # dv/dt calculation
                dvdt_dch = np.diff(cycle_data_dch['Voltage'].values) / np.diff(cycle_data_dch['Time'].values)
                dvdt_ch = np.diff(cycle_data_ch['Voltage'].values) / np.diff(cycle_data_ch['Time'].values)

                # Discharge HIs
                HIs_dch['discharge_time'].append(cycle_data_dch['Time'].values[-1] - cycle_data_dch['Time'].values[0])
                HIs_dch['temperature_std'].append(np.std(cycle_data_dch['Temperature'].values))
                HIs_dch['voltage_std'].append(np.std(cycle_data_dch['Voltage'].values))
                HIs_dch['dv_dt_mean'].append(np.nanmean(dvdt_dch))
                HIs_dch['dv_dt_std'].append(np.nanstd(dvdt_dch))
                HIs_dch['dv_dt_min'].append(np.nanmin(dvdt_dch))
                HIs_dch['current_std'].append(np.std(cycle_data_dch['Current'].values))
                HIs_dch['mean_temp'].append(np.mean(cycle_data_dch['Temperature'].values))
                HIs_dch['max_temp'].append(np.max(cycle_data_dch['Temperature'].values))
                HIs_dch['temp_rise'].append(np.max(cycle_data_dch['Temperature'].values)- np.min(cycle_data_dch['Temperature'].values))
                HIs_dch['starting_voltage'].append(cycle_data_dch['Voltage'].values[0])

                try:
                    HIs_dch['voltage_integral'].append(np.trapz(cycle_data_dch['Voltage'].values, cycle_data_dch['Time'].values))
                except:
                    HIs_dch['voltage_integral'].append(np.nan)

                # Charge HIs
                HIs_ch['charge_time'].append(cycle_data_ch['Time'].values[-1] - cycle_data_ch['Time'].values[0])
                HIs_ch['temperature_std'].append(np.std(cycle_data_ch['Temperature'].values))
                HIs_ch['voltage_std'].append(np.std(cycle_data_ch['Voltage'].values))
                HIs_ch['dv_dt_mean'].append(np.nanmean(dvdt_ch))
                HIs_ch['dv_dt_std'].append(np.nanstd(dvdt_ch))
                HIs_ch['current_std'].append(np.std(cycle_data_ch['Current'].values))
                HIs_ch['mean_temp'].append(np.mean(cycle_data_ch['Temperature'].values))
                HIs_ch['max_temp'].append(np.max(cycle_data_ch['Temperature'].values))
                HIs_ch['temp_rise'].append(np.max(cycle_data_ch['Temperature'].values)- np.min(cycle_data_ch['Temperature'].values))
                HIs_ch['starting_voltage'].append(cycle_data_ch['Voltage'].values[0])

                try:
                    HIs_ch['voltage_integral'].append(np.trapz(cycle_data_ch['Voltage'].values, cycle_data_ch['Time'].values))
                except:
                    HIs_ch['voltage_integral'].append(np.nan)

                # Time to voltage plateau (e.g., 4.2 V during charge)
                plateau_voltage = 4.1
                above_plateau_idx = np.where(cycle_data_ch['Voltage'].values >= plateau_voltage)[0]
                if len(above_plateau_idx) > 0:
                    plateau_time = cycle_data_ch['Time'].values[above_plateau_idx[0]] - cycle_data_ch['Time'].values[0]
                else:
                    plateau_time = np.nan
                HIs_ch['voltage_plateau_time'].append(plateau_time)

                # Voltage at 80% SoC
                try:
                    cap_arr = cycle_data_ch['Capacity'].values
                    max_cap = np.nanmax(cap_arr)
                    cap_80 = 0.8 * max_cap
                    idx_80 = np.argmin(np.abs(cap_arr - cap_80))
                    HIs_ch['voltage_at_80_soc'].append(cycle_data_ch['Voltage'].values[idx_80])
                except:
                    HIs_ch['voltage_at_80_soc'].append(np.nan)


                v_ch = cycle_data_ch['Voltage'].values
                t_ch = cycle_data_ch['Time'].values
                
                sorted_indices = np.argsort(t_ch)
                t_ch_sorted = t_ch[sorted_indices]
                v_ch_sorted = v_ch[sorted_indices]

                kneedle_ch = KneeLocator(
                    t_ch_sorted,
                    v_ch_sorted,
                    S=1,
                    interp_method='polynomial',
                    polynomial_degree = 15,
                    curve='concave',
                    direction='increasing'
                )
                HIs_ch['knee_voltage'].append(kneedle_ch.knee_y if kneedle_ch.knee else np.nan)


                v_dch = cycle_data_dch['Voltage'].values
                t_dch = cycle_data_dch['Time'].values

                sorted_indices = np.argsort(t_dch)
                t_dch_sorted = t_dch[sorted_indices]
                v_dch_sorted = v_dch[sorted_indices]
                
                kneedle_dch = KneeLocator(
                    t_dch_sorted,
                    v_dch_sorted,
                    S=1,
                    interp_method='polynomial',
                    polynomial_degree = 15,
                    curve='concave',
                    direction='decreasing',
                    online=True
                )

                HIs_dch['knee_voltage'].append(kneedle_dch.knee_y if kneedle_dch.knee else np.nan)

                # CC Transition Voltage Calculation
                t_ch = cycle_data_ch['Time'].values
                i_ch = cycle_data_ch['Current'].values
                v_ch = cycle_data_ch['Voltage'].values

                sort_idx = np.argsort(t_ch)
                i_ch_sorted, v_ch_sorted = i_ch[sort_idx], v_ch[sort_idx]

                i_ch_thresh = np.max(i_ch_sorted) - 0.2
                cc_transition_indices_ch = np.where(i_ch_sorted < i_ch_thresh)[0]
                if  len(cc_transition_indices_ch) == 0:
                    HIs_ch['cc_transition_voltage'].append(np.nan)
                else:
                    HIs_ch['cc_transition_voltage'].append(v_ch_sorted[cc_transition_indices_ch[0]])

                t_dch = cycle_data_dch['Time'].values
                i_dch = np.abs(cycle_data_dch['Current'].values)
                v_dch = cycle_data_dch['Voltage'].values
                sort_idx = np.argsort(t_dch)
                i_dch_sorted, v_dch_sorted = i_dch[sort_idx], v_dch[sort_idx]

                i_dch_thresh = np.min(i_dch_sorted) + 0.05
                cc_transition_indices_dch = np.where(i_dch_sorted > i_dch_thresh)[0]
                if len(cc_transition_indices_dch) == 0:
                    HIs_dch['cc_transition_voltage'].append(np.nan)
                else:
                    HIs_dch['cc_transition_voltage'].append(v_dch_sorted[cc_transition_indices_dch][0])

                HIs_dch['knee_transition_diff'].append(HIs_dch['cc_transition_voltage'][-1]-HIs_dch['knee_voltage'][-1])
                HIs_ch['knee_transition_diff'].append(HIs_ch['cc_transition_voltage'][-1]-HIs_ch['knee_voltage'][-1])

        # --- Package original (unequalized) time-series data ---
        data_original_dch = {
            'Voltage': voltage_dch, 'Temperature': temperature_dch,
            'Time': time_dch, 'DTV': DTV_dch, 'DT': DT_dch, 'DV': DV_dch, 'ICA': ICA_dch, 'C_dch':C_dch, 'Current': current_dch
        }
        data_original_ch = {
            'Voltage': voltage_ch, 'Temperature': temperature_ch,
            'Time': time_ch, 'DTV': DTV_ch, 'DT': DT_ch, 'DV': DV_ch, 'ICA': ICA_ch, 'C_ch':C_ch, 'Current': current_ch
        }

        len_eq = 96
        # --- Equalize lengths time-series data ---
        data_eq_ch = {key: self.length_equalizer(val, len_eq = len_eq) for key, val in data_original_ch.items()}
        data_eq_dch = {key: self.length_equalizer(val, len_eq = len_eq) for key, val in data_original_dch.items()}
        
        EoL = {'cycle':None, 'capacity':None}
        # --- Final data structure for return ---
        if self.C_init is not None:
            for cycle, cap in zip(cycles, dch_cap):
                if cap < 0.8 * self.C_init:
                    EoL = {'cycle':cycle, 'capacity':cap}
                    print(f"Cycle {cycle} drops below 80% of initial capacity (C_init = {self.C_init:.3f}) with capacity = {cap:.3f}")
                    break

        return_data = {
            'All': filtered_data,
            'data_dch': data_dch,
            'data_ch': data_ch,
            'Original_dch': data_original_dch,
            'Original_ch': data_original_ch,
            'Equalized_dch': data_eq_dch,
            'Equalized_ch': data_eq_ch,
            'cycles': cycles,
            'dch_cap': dch_cap,
            'ch_cap': ch_cap,
            'HIs_dch': HIs_dch,
            'HIs_ch': HIs_ch,
            'EoL':EoL
        }

        print(f"Cell data loaded from {file_path}")
        return return_data


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def extract_axis_data(self, cycle_data, ax):
        if ax != 'Time':
            return cycle_data[ax].dropna()
        else:
            return cycle_data['Time'].dropna().to_numpy() - cycle_data['Time'].iloc[0]

    def plot_3D(self, ax1, ax2, ax3, start_cycle, end_cycle):
        cmap = plt.get_cmap('plasma')
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        cycles = sorted(self.data['Cycle'].unique())
        filtered_cycles = [cycle for cycle in cycles if start_cycle <= cycle < end_cycle]

        for cycle in filtered_cycles:
            if cycle not in []:
                cycle_data = self.data[(self.data['Cycle'] == cycle)]

                Ax1 = self.extract_axis_data(cycle_data, ax1)
                Ax2 = self.extract_axis_data(cycle_data, ax2)
                Ax3 = self.extract_axis_data(cycle_data, ax3)

                color = cmap(cycle / (end_cycle - 1))
                rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'
                num_ticks = 5
                tick_values = np.linspace(start_cycle, end_cycle, num_ticks + 2).astype(int)

                trace = go.Scatter3d(
                    x=Ax1,
                    y=Ax2,
                    z=Ax3,
                    mode='lines',
                    showlegend=False,
                    line=dict(
                        color=rgb_color,
                        colorbar=dict(
                            title='Cycle Number',
                            tickvals=tick_values,
                            ticks='inside'),
                        cmin=start_cycle,
                        cmax=end_cycle
                    ))
                fig.add_trace(trace)

        fig.update_layout(
            scene=dict(
                xaxis_title=ax1,
                yaxis_title=ax2,
                zaxis_title=ax3
            ))
        fig.show()

    def plot_degradation(self, mode, C_initial=None, marker='o', color='green', markersize=3, label=None, EoL=None, linestyle='-'):
        # mode : ['dch_cap'/'ch_cap', start cycle, reference capacity cycle]
        ch_dch = mode[0]
        start_cycle = mode[1]
        ref_cycle = mode[2]

        Capacity = self.data[ch_dch]

        if C_initial is not None:
            C_n = Capacity / C_initial
        else:
            C_n = Capacity / Capacity[ref_cycle]

        if EoL == None:
            EoL = -1

        cycles = np.array(self.data['cycles']) 
        mask = np.where((cycles > start_cycle) & (cycles < EoL) )[0]

        #plt.plot(cycles[mask], C_n[mask], marker=marker, color=color, markersize=markersize, label=label)
        
        plt.plot(cycles[mask],C_n[mask], marker=marker, color=color, markersize=markersize, label=label, linestyle = linestyle)



    def plot_all(self, cycles_to_plot=[10, 80, 130],
                                dch_his_to_plot=['discharge_time', 'max_temp', 'voltage_std'],
                                ch_his_to_plot=['charge_time', 'max_temp', 'voltage_std'],
                                normalize_his=False,
                                hi_xaxis='capacity'):
        """
        Generates a comprehensive grid of plots to visualize battery data and health indicators.

        Args:
            processed_data (dict): The dictionary returned by a data loading function.
            cycles_to_plot (list, optional): A list of cycle numbers to display in the profile plot.
            dch_his_to_plot (list, optional): A list of strings specifying which discharge HIs to plot.
            ch_his_to_plot (list, optional): A list of strings specifying which charge HIs to plot.
            normalize_his (bool, optional): If True, normalizes HI values to a [0, 1] scale.
            hi_xaxis (str, optional): The x-axis for HI plots. Can be 'cycle' or 'capacity'.
        """
        processed_data = self.data

        # --- Unpack the data ---
        data_dch = processed_data['data_dch']
        data_ch = processed_data['data_ch']
        org_ch = processed_data['Original_ch']
        org_dch = processed_data['Original_dch']
        cycles = np.array(processed_data['cycles'])
        dch_cap = abs(np.array(processed_data['dch_cap']))
        ch_cap = np.array(processed_data['ch_cap'])
        HIs_dch = processed_data['HIs_dch']
        HIs_ch = processed_data['HIs_ch']

        charge_cycles_x_axis = np.arange(1, len(ch_cap) + 1)

        # --- Dynamic Figure Layout ---
        num_hi_plots = len(dch_his_to_plot) + len(ch_his_to_plot)
        num_rows = 1 + math.ceil(num_hi_plots / 2)
        fig = plt.figure(figsize=(18, 7 * num_rows))
        gs = fig.add_gridspec(num_rows, 2)

        hi_plot_count = 0
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'Greys', 'Blues', 
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn']
        
        fig.suptitle('Battery Performance and Health Indicators Analysis', fontsize=20)

        # === Plot 1: Cycle Profiles ===   
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.plasma(np.linspace(0, 1, len(cycles_to_plot)))
        for i, cycle in enumerate(cycles_to_plot):
            cycle_dch_df = data_dch[data_dch['Cycle'] == cycle]
            plt_data_dch = np.diff(cycle_dch_df['Voltage'].values) / np.diff(cycle_dch_df['Time'].values)
            ax1.plot(cycle_dch_df['Time'], cycle_dch_df['Voltage'], color=colors[i], linestyle='-', label=f'V_dch (Cyc {cycle})')
            cycle_ch_df = data_ch[data_ch['Cycle'] == cycle]
            plt_data_ch = np.diff(cycle_ch_df['Voltage'].values) / np.diff(cycle_ch_df['Time'].values)
            ax1.plot(cycle_ch_df['Time'], cycle_ch_df['Voltage'], color=colors[i], linestyle='--')
        ax1.set_title('Voltage Profiles (Solid=DChg, Dashed=Chg)', fontsize=14)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Voltage (V)', fontsize=12)
        ax1.legend()
        ax1.grid(True)

        # === Plot 2: Capacity Degradation ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(cycles, dch_cap, 'o-', label='Discharge Capacity')
        ax2.plot(cycles, ch_cap, 's--', label='Charge Capacity')
        ax2.axhline(y=0.8*dch_cap[3], color='r', linestyle='--', label='EOL Threshold (1.4Ah)')
        ax2.set_title('Capacity Degradation Path', fontsize=14)
        ax2.set_xlabel('Cycle Number', fontsize=12)
        ax2.set_ylabel('Capacity (Ah)', fontsize=12)
        ax2.legend()
        ax2.grid(True)

        # === Dynamic HI Scatter Plots ===
        # Discharge HIs
        for hi_name in dch_his_to_plot:
            row, col = 1 + hi_plot_count // 2, hi_plot_count % 2
            ax_hi = fig.add_subplot(gs[row, col])
            
            if hi_name in HIs_dch:
                x_axis_dch = dch_cap if hi_xaxis == 'capacity' else cycles
                hi_data = np.array(HIs_dch[hi_name])
                if normalize_his:
                    min_val, max_val = hi_data.min(), hi_data.max()
                    hi_data = (hi_data - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(hi_data)
                
                cmap = cmaps[hi_plot_count % len(cmaps)]
                scatter = ax_hi.scatter(x_axis_dch, hi_data, c=cycles, cmap=cmap, s=15, alpha=0.8)
                fig.colorbar(scatter, ax=ax_hi, label='Cycle Number')
                ax_hi.set_title(f'Discharge HI: {hi_name}', fontsize=14)
                ax_hi.set_xlabel('Capacity (Ah)' if hi_xaxis == 'capacity' else 'Cycle Number', fontsize=12)
                ax_hi.set_ylabel('Normalized Value' if normalize_his else 'HI Value', fontsize=12)
                ax_hi.grid(True)
                if hi_xaxis == 'capacity':
                    ax_hi.invert_xaxis()
            hi_plot_count += 1
            
        # Charge HIs
        for hi_name in ch_his_to_plot:
            row, col = 1 + hi_plot_count // 2, hi_plot_count % 2
            ax_hi = fig.add_subplot(gs[row, col])

            if hi_name in HIs_ch:
                x_axis_ch = ch_cap if hi_xaxis == 'capacity' else charge_cycles_x_axis
                hi_data = np.array(HIs_ch[hi_name])
                if normalize_his:
                    min_val, max_val = hi_data.min(), hi_data.max()
                    hi_data = (hi_data - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(hi_data)
                
                cmap = cmaps[hi_plot_count % len(cmaps)]
                scatter = ax_hi.scatter(x_axis_ch, hi_data, c=charge_cycles_x_axis, cmap=cmap, s=15, alpha=0.8)
                fig.colorbar(scatter, ax=ax_hi, label='Cycle Number')
                ax_hi.set_title(f'Charge HI: {hi_name}', fontsize=14)
                ax_hi.set_xlabel('Capacity (Ah)' if hi_xaxis == 'capacity' else 'Cycle Number', fontsize=12)
                ax_hi.set_ylabel('Normalized Value' if normalize_his else 'HI Value', fontsize=12)
                ax_hi.grid(True)
                if hi_xaxis == 'capacity':
                    ax_hi.invert_xaxis()
            hi_plot_count += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        return cycle_ch_df